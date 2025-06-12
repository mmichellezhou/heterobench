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

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def softmax(input_0):
  exp_input_0 = np.exp(input_0)
  sum_total_0 = np.sum(exp_input_0)
  output_0 = exp_input_0 / sum_total_0
  return output_0
*/

void softmax(double *softmax_input, double *exp_results, double *softmax_output, int size) 
{
  // Part 1: Compute exp(input) and sum of exp(input)
  double sum_total_0 = 0;

  // Unroll factor for the first loop. This helps reduce loop overhead
  // and exposes more independent operations for the CPU's out-of-order execution.
  // Using multiple scalar accumulators (local_sum) helps break the dependency chain
  // for the sum, allowing additions to proceed in parallel.
  const int unroll_factor = 8; 
  double local_sum[unroll_factor] = {0.0}; 

  int i = 0;
  // Process elements in chunks of 'unroll_factor'
  for (; i + unroll_factor <= size; i += unroll_factor) {
    // Compute exp for each element and accumulate into separate local sums.
    // The compiler is relied upon to potentially auto-vectorize these 'exp' calls
    // if a vectorized math library (e.g., libmvec) is available and enabled
    // via compiler flags like -ffast-math -march=native.
    exp_results[i] = exp(softmax_input[i]);
    local_sum[0] += exp_results[i];

    exp_results[i+1] = exp(softmax_input[i+1]);
    local_sum[1] += exp_results[i+1];

    exp_results[i+2] = exp(softmax_input[i+2]);
    local_sum[2] += exp_results[i+2];

    exp_results[i+3] = exp(softmax_input[i+3]);
    local_sum[3] += exp_results[i+3];

    exp_results[i+4] = exp(softmax_input[i+4]);
    local_sum[4] += exp_results[i+4];

    exp_results[i+5] = exp(softmax_input[i+5]);
    local_sum[5] += exp_results[i+5];

    exp_results[i+6] = exp(softmax_input[i+6]);
    local_sum[6] += exp_results[i+6];

    exp_results[i+7] = exp(softmax_input[i+7]);
    local_sum[7] += exp_results[i+7];
  }

  // Sum up the local accumulators to get the final sum_total_0
  for (int j = 0; j < unroll_factor; ++j) {
    sum_total_0 += local_sum[j];
  }

  // Handle any remaining elements (tail) that didn't fit into the unrolled loop
  for (; i < size; i++) {
    exp_results[i] = exp(softmax_input[i]);
    sum_total_0 += exp_results[i];
  }

  // Part 2: Compute softmax_output = exp_results / sum_total_0
  // This loop is highly amenable to SIMD vectorization.
  // We use AVX2 intrinsics for double-precision floating-point operations.
  
  // Broadcast sum_total_0 into an AVX2 register so it can be used for element-wise division
  __m256d sum_total_vec = _mm256_set1_pd(sum_total_0);
  const int vec_size = 4; // A __m256d register holds 4 doubles

  i = 0;
  // Process elements in chunks of 'vec_size' (4 doubles) using AVX2 intrinsics
  for (; i + vec_size <= size; i += vec_size) {
    // Load 4 double values from exp_results (unaligned load is safe with _mm256_loadu_pd)
    __m256d exp_res_vec = _mm256_loadu_pd(&exp_results[i]);
    
    // Perform element-wise division: exp_res_vec / sum_total_vec
    __m256d output_vec = _mm256_div_pd(exp_res_vec, sum_total_vec);
    
    // Store the 4 resulting double values into softmax_output (unaligned store is safe)
    _mm256_storeu_pd(&softmax_output[i], output_vec);
  }

  // Handle any remaining elements (tail) for the division loop
  for (; i < size; i++) {
    softmax_output[i] = exp_results[i] / sum_total_0;
  }
}