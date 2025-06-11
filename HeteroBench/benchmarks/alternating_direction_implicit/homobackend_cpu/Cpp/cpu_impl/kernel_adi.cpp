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
  // The parameters 'tsteps' and 'n' are passed, but the original code uses
  // the global/macro definitions 'TSTEPS' and 'N'. We adhere to the original
  // behavior by continuing to use 'TSTEPS' and 'N'.

  {
    int c0; // Loop variable for time steps
    int c2; // Loop variable for row/column index (outer loop in original, inner in optimized)
    int c8; // Loop variable for inner dimension (inner loop in original, outer in optimized)
    
    for (c0 = 0; c0 <= TSTEPS; c0++) {
      // Phase 1: Operations along rows (horizontal sweeps)
      // These loops already exhibit good cache locality due to row-major access.
      // The provided code already includes key optimizations:
      // 1. Replaced division with multiplication by reciprocal (1.0 / divisor) to potentially improve throughput.
      // 2. Common subexpression elimination for repeated calculations within inner loops.
      // Further direct SIMD vectorization for these loops is limited by inherent recurrence relations
      // (e.g., B[c2][c8] depends on B[c2][c8-1]), which require sequential computation.
      for (c2 = 0; c2 <= N - 1; c2++) { // Iterate over rows
        // Forward sweep for B
        for (c8 = 1; c8 <= N - 1; c8++) {
          const double inv_B_prev = 1.0 / B[c2][c8 - 1]; // Compute reciprocal once per iteration
          const double A_sq = A[c2][c8] * A[c2][c8];     // Common subexpression
          B[c2][c8] = B[c2][c8] - A_sq * inv_B_prev;
        }
        // Forward sweep for X
        for (c8 = 1; c8 <= N - 1; c8++) {
          // B[c2][c8-1] might have been updated in the previous loop, so recompute reciprocal.
          const double inv_B_prev = 1.0 / B[c2][c8 - 1]; 
          const double term_X_A = X[c2][c8 - 1] * A[c2][c8]; // Common subexpression
          X[c2][c8] = X[c2][c8] - term_X_A * inv_B_prev;
        }
        // Backward sweep for X
        for (c8 = N - 2; c8 >= 1; c8--) { // c8 now represents the actual column index
          const double inv_B_curr = 1.0 / B[c2][c8]; // Compute reciprocal once per iteration
          const double term_X_A = X[c2][c8 - 1] * A[c2][c8 - 1]; // Common subexpression
          X[c2][c8] = (X[c2][c8] - term_X_A) * inv_B_curr;
        }
      }
      // Final division for the last column of X
      // This loop accesses elements in a column (e.g., X[0][N-1], X[1][N-1], ...).
      // This results in strided memory access, which is generally not cache-friendly
      // and hinders efficient auto-vectorization without specialized gather/scatter instructions.
      // Due to constraints on maintaining array access patterns, this structure is preserved.
      for (c2 = 0; c2 <= N - 1; c2++) {
        const double inv_B_last = 1.0 / B[c2][N - 1]; // Compute reciprocal once per iteration
        X[c2][N - 1] = X[c2][N - 1] * inv_B_last;
      }

      // Phase 2: Operations along columns (vertical sweeps)
      // The original implementation had column-major access patterns (e.g., B[c8][c2] with c2 outer, c8 inner),
      // leading to poor cache performance.
      // The provided code already swapped the loop order (c8 outer, c2 inner) to achieve row-major access
      // for the inner loop, significantly improving cache locality and enabling
      // better auto-vectorization by the compiler for the inner loop's *scalar* operations.
      // The data dependencies (e.g., B[c8][c2] depends on B[c8-1][c2]) are preserved
      // because 'c8' (the row index) is now the outer loop, ensuring B[c8-1][c2]
      // is computed before B[c8][c2].
      // The provided code already includes key optimizations:
      // 1. Replaced division with multiplication by reciprocal (1.0 / divisor).
      // 2. Common subexpression elimination for repeated calculations within inner loops.
      // The inner 'c2' loops are now contiguous and independent, making them ideal for vectorization.
      // Compiler hints (`_Pragma("GCC ivdep")`) are added to encourage auto-vectorization.

      // Forward sweep for B (optimized loop order)
      for (c8 = 1; c8 <= N - 1; c8++) { // Iterate over rows (outer loop)
        _Pragma("GCC ivdep") // Hint to the compiler for vectorization
        for (c2 = 0; c2 <= N - 1; c2++) { // Iterate over columns (inner loop, contiguous access)
          const double inv_B_prev_col = 1.0 / B[c8 - 1][c2]; // Compute reciprocal once per iteration
          const double A_sq = A[c8][c2] * A[c8][c2];         // Common subexpression
          B[c8][c2] = B[c8][c2] - A_sq * inv_B_prev_col;
        }
      }
      // Forward sweep for X (optimized loop order)
      for (c8 = 1; c8 <= N - 1; c8++) { // Iterate over rows (outer loop)
        _Pragma("GCC ivdep") // Hint to the compiler for vectorization
        for (c2 = 0; c2 <= N - 1; c2++) { // Iterate over columns (inner loop, contiguous access)
          // B[c8-1][c2] might have been updated in the previous loop, so recompute reciprocal.
          const double inv_B_prev_col = 1.0 / B[c8 - 1][c2]; 
          const double term_X_A = X[c8 - 1][c2] * A[c8][c2]; // Common subexpression
          X[c8][c2] = X[c8][c2] - term_X_A * inv_B_prev_col;
        }
      }
      // Backward sweep for X (optimized loop order)
      for (c8 = N - 2; c8 >= 1; c8--) { // Iterate over rows (outer loop, backward)
        _Pragma("GCC ivdep") // Hint to the compiler for vectorization
        for (c2 = 0; c2 <= N - 1; c2++) { // Iterate over columns (inner loop, contiguous access)
          const double inv_B_curr_col = 1.0 / B[c8][c2]; // Compute reciprocal once per iteration
          const double term_X_A = X[c8 - 1][c2] * A[c8 - 1][c2]; // Common subexpression
          X[c8][c2] = (X[c8][c2] - term_X_A) * inv_B_curr_col;
        }
      }

      // Final division for the last row of X
      // This loop is highly vectorizable as it's an element-wise operation on contiguous data.
      _Pragma("GCC ivdep") // Hint to the compiler for vectorization
      for (c2 = 0; c2 <= N - 1; c2++) {
        const double inv_B_last_row = 1.0 / B[N - 1][c2]; // Compute reciprocal once per iteration
        X[N - 1][c2] = X[N - 1][c2] * inv_B_last_row;
      }
    }
  }
}