/*
 * (C) Copyright [2024] Hewlett Packard Enterprise Development LP
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
 
#include "cpu_impl.h"

using namespace std;

void kernel_adi(int tsteps,int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  // The original code uses N and TSTEPS macros (presumably defined in cpu_impl.h),
  // not the function arguments `n` and `tsteps`. We adhere to this behavior.
  
  // The outer loop iterates over time steps.
  for (int c0 = 0; c0 <= TSTEPS; c0++) {
    // Phase 1: Horizontal (row-wise) computations
    // The outer loop iterates over rows (c2).
    // Access patterns B[c2][c8], A[c2][c8], X[c2][c8] are cache-friendly (row-major).
    for (int c2 = 0; c2 <= N - 1; c2++) {
      // Loop 1: B update (forward sweep)
      // This loop has a loop-carried dependency (recurrence) on B[c2][c8 - 1].
      // The `omp simd` pragma hints to the compiler to vectorize this loop.
      // While full vectorization of the recurrence might not be possible,
      // compilers can often vectorize independent parts (e.g., A[c2][c8] * A[c2][c8])
      // or apply partial vectorization/unrolling.
      _Pragma("omp simd")
      for (int c8 = 1; c8 <= N - 1; c8++) {
        B[c2][c8] = B[c2][c8] - A[c2][c8] * A[c2][c8] / B[c2][c8 - 1];
      }

      // Loop 2: X update (forward sweep)
      // This loop also has loop-carried dependencies on X[c2][c8 - 1] and B[c2][c8 - 1].
      _Pragma("omp simd")
      for (int c8 = 1; c8 <= N - 1; c8++) {
        X[c2][c8] = X[c2][c8] - X[c2][c8 - 1] * A[c2][c8] / B[c2][c8 - 1];
      }

      // Loop 3: X update (backward sweep)
      // This loop processes elements in reverse order (from N-2 down to 1).
      // It also has loop-carried dependencies.
      _Pragma("omp simd")
      for (int c8 = 0; c8 <= N - 3; c8++) {
        X[c2][N - c8 - 2] = (X[c2][N - 2 - c8] - X[c2][N - 2 - c8 - 1] * A[c2][N - c8 - 3]) / B[c2][N - 3 - c8];
      }
    }

    // Loop 4: X update for the last column
    // This loop is perfectly parallel across `c2` iterations, making it highly suitable for vectorization.
    _Pragma("omp simd")
    for (int c2 = 0; c2 <= N - 1; c2++) {
      X[c2][N - 1] = X[c2][N - 1] / B[c2][N - 1];
    }

    // Phase 2: Vertical (column-wise) computations
    // The outer loop iterates over columns (c2).
    // Accessing B[c8][c2], A[c8][c2], X[c8][c2] results in strided memory access for C/C++ (row-major arrays).
    // This can lead to poor cache performance. Due to the requirement to "maintain exact same array access patterns",
    // we cannot reorder loops or change array indexing to improve cache locality for this phase.
    // However, vectorization hints are still provided.
    for (int c2 = 0; c2 <= N - 1; c2++) {
      // Loop 5: B update (forward sweep, strided access)
      // Recurrence on B[c8 - 1][c2].
      _Pragma("omp simd")
      for (int c8 = 1; c8 <= N - 1; c8++) {
        B[c8][c2] = B[c8][c2] - A[c8][c2] * A[c8][c2] / B[c8 - 1][c2];
      }

      // Loop 6: X update (forward sweep, strided access)
      // Recurrence on X[c8 - 1][c2] and B[c8 - 1][c2].
      _Pragma("omp simd")
      for (int c8 = 1; c8 <= N - 1; c8++) {
        X[c8][c2] = X[c8][c2] - X[c8 - 1][c2] * A[c8][c2] / B[c8 - 1][c2];
      }

      // Loop 7: X update (backward sweep, strided access)
      // Recurrence, similar to Loop 3 but operating along columns.
      _Pragma("omp simd")
      for (int c8 = 0; c8 <= N - 3; c8++) {
        X[N - 2 - c8][c2] = (X[N - 2 - c8][c2] - X[N - c8 - 3][c2] * A[N - 3 - c8][c2]) / B[N - 2 - c8][c2];
      }
    }

    // Loop 8: X update for the last row
    // This loop is perfectly parallel across `c2`, but involves strided memory access.
    _Pragma("omp simd")
    for (int c2 = 0; c2 <= N - 1; c2++) {
      X[N - 1][c2] = X[N - 1][c2] / B[N - 1][c2];
    }
  }
}