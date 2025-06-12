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
#include <math.h>
#include <immintrin.h> // Required for AVX intrinsics

// Using namespace std; is kept for consistency with the original code's style.
using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def conv2d(input, kernel, bias, stride, padding):
  input_padded = pad_input(input, padding)
  kernel_height, kernel_width = kernel.shape
  output_height = (input_padded.shape[0] - kernel_height) // stride + 1
  output_width = (input_padded.shape[1] - kernel_width) // stride + 1
  conv2d_output = np.zeros((output_height, output_width))
  for i in range(0, output_height):
    for j in range(0, output_width):
      region = input_padded[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width]
      conv2d_output[i, j] = np.sum(region * kernel) + bias
  return conv2d_output
*/

void conv2d(double *conv2d_input, double *conv2d_kernel, double *input_padded, double conv2d_bias, int stride, int padding, int input_h, int input_w, int kernel_h, int kernel_w, double *conv2d_output)
{
  // The pad_input function is assumed to be defined and implemented elsewhere (e.g., in cpu_impl.h).
  // Its performance is not part of this optimization task.
  pad_input(conv2d_input, input_padded, input_h, input_w, padding);

  // Pre-calculate constant dimensions for efficiency.
  // These values are loop invariants for the outer loops.
  const int padded_width = input_w + 2 * padding;
  const int output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
  const int output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

  // Define vector size for AVX (256-bit) doubles.
  // A __m256d vector holds 4 double-precision floating-point numbers.
  const int VEC_SIZE = 4;

  // Iterate over each output pixel (i, j)
  for (int i = 0; i < output_h; i++) {
    // Pre-calculate the base row index for the input_padded region for the current 'i'.
    // This avoids repeated multiplication inside the inner loops.
    const int input_padded_row_base_idx = i * stride;

    for (int j = 0; j < output_w; j++) {
      // Initialize a scalar accumulator for the current output pixel's sum.
      double tmp = 0.0;

      // Pre-calculate the base column index for the input_padded region for the current 'j'.
      const int input_padded_col_base_idx = j * stride;

      // Iterate over kernel height (k)
      for (int k = 0; k < kernel_h; k++) {
        // Calculate the starting memory address for the current row of the input region
        // and the current row of the kernel.
        // Using direct pointers improves memory access efficiency by reducing address calculations.
        const double* current_input_row_ptr = input_padded + (input_padded_row_base_idx + k) * padded_width + input_padded_col_base_idx;
        const double* current_kernel_row_ptr = conv2d_kernel + k * kernel_w;

        // Initialize a vector accumulator for the sum of products for the current kernel row (k).
        // This will accumulate 4 products simultaneously.
        __m256d row_sum_vec = _mm256_setzero_pd();

        int l = 0;
        // Vectorized loop for kernel width (l) using AVX intrinsics.
        // Process VEC_SIZE (4) doubles at a time.
        // _mm256_loadu_pd is used for unaligned loads, which is safer as array alignment is not guaranteed.
        for (; l + VEC_SIZE <= kernel_w; l += VEC_SIZE) {
          __m256d input_vec = _mm256_loadu_pd(current_input_row_ptr + l);
          __m256d kernel_vec = _mm256_loadu_pd(current_kernel_row_ptr + l);

          // Perform fused multiply-add: row_sum_vec = (input_vec * kernel_vec) + row_sum_vec.
          // This instruction combines multiplication and addition into a single operation,
          // which is highly efficient on CPUs with FMA support.
          row_sum_vec = _mm256_fmadd_pd(input_vec, kernel_vec, row_sum_vec);
        }

        // Horizontal sum of the vector accumulator (row_sum_vec).
        // This sequence efficiently sums the 4 doubles in the __m256d register into a single scalar.
        // 1. Swap 128-bit lanes: [A,B,C,D] -> [C,D,A,B]
        __m256d sum_vec_perm = _mm256_permute2f128_pd(row_sum_vec, row_sum_vec, 1);
        // 2. Add original and swapped: [A+C, B+D, C+A, D+B]
        row_sum_vec = _mm256_add_pd(row_sum_vec, sum_vec_perm);
        // 3. Get the lower 128-bit lane: [A+C, B+D]
        __m128d sum_vec_128 = _mm256_castpd256_pd128(row_sum_vec);
        // 4. Horizontal add within 128-bit: [ (A+C)+(B+D), (A+C)+(B+D) ]
        __m128d sum_vec_hadd = _mm_hadd_pd(sum_vec_128, sum_vec_128);
        // 5. Extract the first double (which contains the total sum) and add to 'tmp'.
        tmp += _mm_cvtsd_f64(sum_vec_hadd);

        // Remainder loop for kernel width (l).
        // This loop handles any remaining elements if kernel_w is not a multiple of VEC_SIZE.
        for (; l < kernel_w; ++l) {
          tmp += current_input_row_ptr[l] * current_kernel_row_ptr[l];
        }
      }
      // Add the bias and store the final result in the output array.
      conv2d_output[i * output_w + j] = tmp + conv2d_bias;
    }
  }
}