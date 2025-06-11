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

// The 'using namespace std;' directive was present in the original main.cpp,
// but is not necessary within the scope of this function and is omitted for clarity.

void init_array(int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  // The original code included an unnecessary outer block and commented-out variable declarations.
  // These have been removed for cleaner code.

  if (n >= 1) {
    // Optimization: Strength reduction for division by 'n'.
    // The division by 'n' is common to all three assignments in the inner loop.
    // Precomputing '1.0 / n' once outside the loops replaces three divisions per inner loop iteration
    // with a single multiplication, which is significantly faster.
    const double inv_n = 1.0 / n;

    // Loop over rows (c1)
    for (int c1 = 0; c1 < n; c1++) {
      // Optimization: Strength reduction for casting 'c1' to double.
      // The value of 'c1' is constant within the inner loop. Casting it to double
      // once per outer loop iteration avoids redundant integer-to-float conversions.
      const double dc1 = (double)c1;

      // Loop over columns (c2)
      // This inner loop is highly amenable to auto-vectorization by modern compilers (e.g., GCC, Clang, Intel ICC).
      // 1. Memory Access Pattern: The arrays are accessed in row-major order (X[c1][c2], A[c1][c2], B[c1][c2])
      //    with 'c2' being the innermost varying index. This results in contiguous memory accesses,
      //    which is highly cache-friendly and ideal for SIMD (Single Instruction, Multiple Data) operations.
      // 2. Independent Operations: The calculations for each (c1, c2) pair are independent of other pairs,
      //    allowing for parallel execution of multiple elements using SIMD registers.
      // 3. Compiler Optimizations: With appropriate compiler flags (e.g., -O3 -march=native or -O3 -mavx),
      //    compilers can automatically generate SIMD instructions (like AVX for doubles) and perform
      //    loop unrolling, instruction scheduling, and common subexpression elimination.
      //    Relying on auto-vectorization is generally preferred for maintainability and robustness
      //    unless profiling indicates a specific bottleneck that requires manual intrinsics.
      for (int c2 = 0; c2 < n; c2++) {
        // Original expressions:
        // X[c1][c2] = (((double )c1) * (c2 + 1) + 1) / n;
        // A[c1][c2] = (((double )c1) * (c2 + 2) + 2) / n;
        // B[c1][c2] = (((double )c1) * (c2 + 3) + 3) / n;

        // Optimized expressions using precomputed values (dc1 and inv_n).
        // This simplifies the arithmetic and exposes opportunities for the compiler
        // to apply further optimizations like Fused Multiply-Add (FMA) instructions
        // if supported by the target architecture.
        X[c1][c2] = (dc1 * (c2 + 1) + 1) * inv_n;
        A[c1][c2] = (dc1 * (c2 + 2) + 2) * inv_n;
        B[c1][c2] = (dc1 * (c2 + 3) + 3) * inv_n;
      }
    }
  }
}