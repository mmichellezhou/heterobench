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
#include <iostream>
#include <immintrin.h> // For AVX intrinsics (__m256d, _mm256_*, etc.)

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
  def dot_add(x, W, b):
    mm = np.dot(x, W) + b
    return mm
*/

void dot_add(double *dot_add_input_x, double *dot_add_input_W, double *dot_add_input_b, double *dot_add_output, int x_h, int x_w, int W_h, int W_w) {
  // This optimized implementation focuses on single-threaded performance improvements by:
  // 1. Changing the loop order for the matrix multiplication to improve cache locality.
  //    The original i-j-k loop order results in strided access to W (column-wise access for row-major W),
  //    leading to poor cache performance. The optimized i-k-j order ensures sequential, cache-friendly
  //    accesses to both W and the output matrix.
  // 2. Merging the bias addition into the matrix multiplication phase. Instead of a separate loop,
  //    the output matrix is first initialized with the bias values, and then the matrix product
  //    is accumulated onto it. This avoids a second full pass over the output matrix.
  // 3. Applying SIMD (Single Instruction, Multiple Data) vectorization using AVX intrinsics.
  //    The innermost loops (for 'j') are vectorized to process multiple `double` elements
  //    simultaneously, leveraging modern CPU capabilities.

  // Phase 1: Initialize dot_add_output with bias values (dot_add_output = b)
  // This loop iterates row by row for dot_add_output and sequentially for dot_add_input_b,
  // ensuring cache-friendly access patterns.
  for (int i = 0; i < x_h; i++) {
    int j = 0;
    // Vectorized part for bias initialization (AVX processes 4 doubles at a time with __m256d)
    // _mm256_loadu_pd and _mm256_storeu_pd are used for unaligned memory access safety.
    // If memory is guaranteed to be 32-byte aligned (e.g., via posix_memalign),
    // _mm256_load_pd and _mm256_store_pd could be used for potentially better performance.
    for (; j + 3 < W_w; j += 4) {
      _mm256d b_vec = _mm256_loadu_pd(&dot_add_input_b[j]);
      _mm256_storeu_pd(&dot_add_output[i * W_w + j], b_vec);
    }
    // Handle remaining elements (if W_w is not a multiple of 4)
    for (; j < W_w; j++) {
      dot_add_output[i * W_w + j] = dot_add_input_b[j];
    }
  }

  // Phase 2: Perform matrix multiplication (x * W) and add the result to the pre-initialized output.
  // The loop order is changed from the original (i-j-k) to (i-k-j) for better cache locality.
  // This ensures sequential access to rows of W and output, improving data reuse.
  // Note: For a valid matrix multiplication (x * W), x_w (x.shape[1]) must be equal to W_h (W.shape[0]).
  // The 'k' loop iterates up to x_w, implicitly using the inner dimension.
  for (int i = 0; i < x_h; i++) {
    for (int k = 0; k < x_w; k++) { // Loop over the inner dimension (x_w, which is equivalent to W_h)
      // Load x[i][k] once for the entire inner 'j' loop. This value is constant for the inner loop.
      double val_x_ik = dot_add_input_x[i * x_w + k];
      // Broadcast this scalar value into an AVX vector for element-wise multiplication.
      _mm256d x_val_vec = _mm256_set1_pd(val_x_ik);

      int j = 0;
      // Vectorized part for matrix multiplication (AVX processes 4 doubles at a time)
      for (; j + 3 < W_w; j += 4) {
        // Load a segment of W's k-th row (W[k][j...j+3]). This is a cache-friendly, sequential access.
        _mm256d W_vec = _mm256_loadu_pd(&dot_add_input_W[k * W_w + j]);
        // Load a segment of output's i-th row (output[i][j...j+3]). This is also cache-friendly.
        _mm256d output_vec = _mm256_loadu_pd(&dot_add_output[i * W_w + j]);

        // Perform element-wise multiplication: x_val_vec * W_vec
        _mm256d mul_res = _mm256_mul_pd(x_val_vec, W_vec);
        // Perform element-wise addition: output_vec + mul_res
        // Compilers often translate this pattern into Fused Multiply-Add (FMA) instructions
        // (e.g., _mm256_fmadd_pd) if the target architecture supports AVX2/FMA, which is more efficient.
        _mm256d add_res = _mm256_add_pd(output_vec, mul_res);

        // Store the updated segment back to output.
        _mm256_storeu_pd(&dot_add_output[i * W_w + j], add_res);
      }
      // Handle remaining elements (if W_w is not a multiple of 4)
      for (; j < W_w; j++) {
        dot_add_output[i * W_w + j] += val_x_ik * dot_add_input_W[k * W_w + j];
      }
    }
  }
}