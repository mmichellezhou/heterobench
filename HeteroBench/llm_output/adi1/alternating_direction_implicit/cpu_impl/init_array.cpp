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

void init_array(int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  // The original code included an unnecessary extra scope block and commented out variables.
  // These have been removed for cleaner and more idiomatic C++ code.
  
  // The condition `n >= 1` is crucial for correctness, as it prevents division by zero
  // and ensures loops run only for valid dimensions.
  if (n >= 1) {
    // Optimization: Strength Reduction
    // Precompute `1.0 / n` once outside the loops. This transforms `n` divisions per element
    // into a single multiplication per element, which is significantly faster on modern CPUs.
    const double inv_n = 1.0 / static_cast<double>(n);

    // The outer loop iterates through rows. Using `c1 < n` is idiomatic C++ and equivalent
    // to the original `c1 <= n + -1`.
    for (int c1 = 0; c1 < n; ++c1) {
      // Optimization: Strength Reduction
      // Precompute `(double)c1` once per outer loop iteration.
      // This avoids redundant integer-to-double conversions and multiplications within the inner loop.
      const double dc1 = static_cast<double>(c1);

      // The inner loop iterates through columns. This loop is the primary target for
      // compiler auto-vectorization (SIMD instructions).
      // The memory access pattern `[c1][c2]` is row-major, which is optimal for cache
      // performance in C/C++ (contiguous memory access).
      // Modern compilers (e.g., GCC, Clang, Intel) with `-O3` and appropriate architecture flags
      // (like `-march=native` or `-mavx2`) are highly effective at auto-vectorizing such loops
      // due to the independent arithmetic operations and predictable memory access.
      for (int c2 = 0; c2 < n; ++c2) {
        // Original expressions:
        // X[c1][c2] = (((double )c1) * (c2 + 1) + 1) / n;
        // A[c1][c2] = (((double )c1) * (c2 + 2) + 2) / n;
        // B[c1][c2] = (((double )c1) * (c2 + 3) + 3) / n;

        // Optimized expressions using the precomputed `dc1` and `inv_n`.
        // `static_cast<double>(c2 + K)` ensures the integer sum `c2 + K` is converted to
        // double before multiplication, maintaining functional equivalence to the original.
        // Explicit `1.0`, `2.0`, `3.0` as doubles ensures floating-point arithmetic.
        // Compilers can often generate Fused Multiply-Add (FMA) instructions for these patterns,
        // further improving throughput.
        X[c1][c2] = (dc1 * static_cast<double>(c2 + 1) + 1.0) * inv_n;
        A[c1][c2] = (dc1 * static_cast<double>(c2 + 2) + 2.0) * inv_n;
        B[c1][c2] = (dc1 * static_cast<double>(c2 + 3) + 3.0) * inv_n;
      }
    }
  }
}