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
#include <immintrin.h> // Required for AVX (Advanced Vector Extensions) intrinsics

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
  // Step 1: Pad the input array. This function is called once and its implementation
  // is assumed to be external and not part of this optimization task.
  pad_input(conv2d_input, input_padded, input_h, input_w, padding);

  // Step 2: Pre-calculate dimensions and strides.
  // These calculations are moved outside the main loops to avoid redundant computations.
  const int input_padded_w = input_w + 2 * padding;
  const int output_h = (input_h + 2 * padding - kernel_h) / stride + 1;
  const int output_w = (input_w + 2 * padding - kernel_w) / stride + 1;

  // Pre-calculate row strides for efficient pointer arithmetic.
  const int input_padded_row_stride = input_padded_w;
  const int kernel_row_stride = kernel_w;
  const int output_row_stride = output_w;

  // Step 3: Vectorization setup for AVX (256-bit registers for doubles).
  // A __m256d register can hold 4 double-precision floating-point values.
  const int VEC_SIZE = 4;

  // Outer loops iterate over the output feature map dimensions (height and width).
  for (int i = 0; i < output_h; i++) {
    // Calculate the base pointer for the current row in the padded input.
    // This avoids repeated multiplication (i * stride * input_padded_row_stride) in the inner loop.
    // Using long long for pointer arithmetic to prevent potential overflow with large dimensions.
    const double* input_padded_base_row_i = input_padded + (long long)i * stride * input_padded_row_stride;

    for (int j = 0; j < output_w; j++) {
      double tmp_scalar_sum = 0.0; // Accumulator for scalar operations (remainder elements)
      __m256d tmp_vector_sum = _mm256_setzero_pd(); // Accumulator for vector operations

      // Calculate the base pointer for the current region in the padded input for this output pixel (i, j).
      const double* current_input_padded_region_base = input_padded_base_row_i + (long long)j * stride;

      // Inner loops iterate over the kernel dimensions (height and width).
      for (int k = 0; k < kernel_h; k++) {
        // Pointers to the current row in the input_padded region and kernel.
        const double* input_ptr = current_input_padded_region_base + (long long)k * input_padded_row_stride;
        const double* kernel_ptr = conv2d_kernel + (long long)k * kernel_row_stride;

        // Vectorized loop: Process kernel_w elements using AVX intrinsics.
        // This loop processes VEC_SIZE (4) elements at a time.
        int l = 0;
        for (; l + VEC_SIZE <= kernel_w; l += VEC_SIZE) {
          // Load 4 double-precision values from input_padded and kernel.
          // _mm256_loadu_pd is used for unaligned memory access, which is generally safer
          // as the input arrays might not be 32-byte aligned.
          __m256d input_vec = _mm256_loadu_pd(input_ptr + l);
          __m256d kernel_vec = _mm256_loadu_pd(kernel_ptr + l);

          // Perform element-wise multiplication (input_vec * kernel_vec) and
          // accumulate the results into tmp_vector_sum.
          tmp_vector_sum = _mm256_add_pd(tmp_vector_sum, _mm256_mul_pd(input_vec, kernel_vec));
        }

        // Scalar loop: Handle any remaining elements if kernel_w is not a multiple of VEC_SIZE.
        for (; l < kernel_w; l++) {
          tmp_scalar_sum += input_ptr[l] * kernel_ptr[l];
        }
      }

      // Step 4: Horizontal sum of the vector accumulator.
      // This sums the 4 double-precision values in tmp_vector_sum into a single scalar double.
      // 1. _mm256_hadd_pd sums adjacent pairs: [v0+v1, v2+v3, v0+v1, v2+v3]
      __m256d hsum_vec = _mm256_hadd_pd(tmp_vector_sum, tmp_vector_sum);
      // 2. Extract the first 128-bit lane, which contains [v0+v1, v2+v3].
      __m128d sum_low128 = _mm256_extractf128_pd(hsum_vec, 0);
      // 3. Add the two doubles from the 128-bit lane to the scalar sum.
      tmp_scalar_sum += _mm_cvtsd_f64(sum_low128); // Extracts and adds (v0+v1)
      tmp_scalar_sum += _mm_cvtsd_f64(_mm_shuffle_pd(sum_low128, sum_low128, 0x01)); // Extracts and adds (v2+v3)

      // Step 5: Add the bias and store the final result in the output array.
      conv2d_output[(long long)i * output_row_stride + j] = tmp_scalar_sum + conv2d_bias;
    }
  }
}