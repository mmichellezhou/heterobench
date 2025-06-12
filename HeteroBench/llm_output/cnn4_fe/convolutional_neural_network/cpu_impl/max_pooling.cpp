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
#include <math.h>        // For original code's max, though std::max is preferred
#include <limits>        // For std::numeric_limits
#include <immintrin.h>   // For AVX intrinsics (e.g., __m256d, _mm256_loadu_pd, _mm256_max_pd)
#include <algorithm>     // For std::max

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def max_pooling(input, pool_size, pool_stride=2):
  output_height = (input.shape[0] - pool_size) // pool_stride + 1
  output_width = (input.shape[1] - pool_size) // pool_stride + 1
  output = np.zeros((output_height, output_width))
  for i in range(0, output_height):
    for j in range(0, output_width):
      region = input[i*pool_stride:i*pool_stride+pool_size, j*pool_stride:j*pool_stride+pool_size]
      output[i, j] = np.max(region)
  return output
*/

void max_pooling(double *max_pooling_input, int pool_size, int pool_stride, int input_h, int input_w, double *max_pooling_output) {
  int output_h = (input_h - pool_size) / pool_stride + 1;
  int output_w = (input_w - pool_size) / pool_stride + 1;

  // Define vector width for doubles (4 doubles for AVX2)
  const int VEC_DOUBLES = 4; // __m256d holds 4 doubles

  for (int i = 0; i < output_h; i++) {
    for (int j = 0; j < output_w; j++) {
      // Initialize scalar max_val to the lowest possible double value.
      // This ensures correctness for regions containing negative numbers,
      // matching numpy's behavior for np.max.
      double max_val = std::numeric_limits<double>::lowest(); 

      // Calculate the base address for the current pooling window's top-left corner.
      // This pointer points to the element max_pooling_input[ (i * pool_stride) * input_w + (j * pool_stride) ].
      // Using static_cast<long long> to prevent potential integer overflow for large dimensions.
      double *window_base_ptr = max_pooling_input + (static_cast<long long>(i) * pool_stride) * input_w + (static_cast<long long>(j) * pool_stride);

      for (int k = 0; k < pool_size; k++) {
        // Pointer to the start of the k-th row within the current pooling window.
        // This pointer points to max_pooling_input[ (i * pool_stride + k) * input_w + (j * pool_stride) ].
        double *current_input_row_ptr = window_base_ptr + static_cast<long long>(k) * input_w;

        // Initialize a vector register for accumulating the maximum value for the current row.
        __m256d row_max_vec = _mm256_set1_pd(std::numeric_limits<double>::lowest()); 

        // Vectorized part for 'l' loop (columns within the window).
        // Process elements in chunks of VEC_DOUBLES (4 doubles for AVX2).
        int l = 0;
        for (; l + VEC_DOUBLES <= pool_size; l += VEC_DOUBLES) {
          // Load 4 doubles from memory into an AVX register (unaligned load is safe and often optimized).
          __m256d input_vec = _mm256_loadu_pd(current_input_row_ptr + l);
          // Perform element-wise maximum operation between the current row_max_vec and the loaded input_vec.
          row_max_vec = _mm256_max_pd(row_max_vec, input_vec);
        }

        // Horizontal reduction of row_max_vec to a single double scalar.
        // This sequence efficiently finds the maximum value within the 4 doubles of the __m256d register.
        // Step 1: Max of the lower 128 bits and the upper 128 bits.
        __m128d v_max_128 = _mm_max_pd(_mm256_castpd256_pd128(row_max_vec), _mm256_extractf128_pd(row_max_vec, 1));
        // Step 2: Max of the two doubles remaining in the 128-bit vector.
        __m128d v_max_final = _mm_max_sd(v_max_128, _mm_shuffle_pd(v_max_128, v_max_128, 1));
        // Step 3: Extract the final scalar maximum value from the lowest element of the 128-bit vector.
        double row_max_from_vec = _mm_cvtsd_f64(v_max_final);

        // Update the overall max_val for the current pooling window with the maximum found in the vectorized part of this row.
        max_val = std::max(max_val, row_max_from_vec);

        // Scalar cleanup loop for remaining elements in 'l' (if pool_size is not a multiple of VEC_DOUBLES).
        for (; l < pool_size; l++) {
          max_val = std::max(max_val, current_input_row_ptr[l]);
        }
      }
      // Store the final maximum value for the current pooling window in the output array.
      max_pooling_output[static_cast<long long>(i) * output_w + j] = max_val;
    }
  }
}