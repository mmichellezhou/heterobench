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
  double sum_total_0 = 0;

  // Optimization 1: Loop Unrolling for the first loop.
  // This reduces loop overhead and exposes more instruction-level parallelism
  // for the CPU's out-of-order execution engine. It can also aid the compiler
  // in generating better SIMD (vectorized) code, especially for the exp function
  // if a vectorized math library (like libmvec) is available and enabled by compiler flags.
  // An unroll factor of 4 is chosen to align with common SIMD register widths for doubles (e.g., AVX-256).
  const int unroll_factor = 4;
  int i = 0;

  // Process elements in chunks of 'unroll_factor'
  for (; i + unroll_factor <= size; i += unroll_factor) {
    // Calculate exp for unrolled elements
    double val0 = exp(softmax_input[i]);
    double val1 = exp(softmax_input[i+1]);
    double val2 = exp(softmax_input[i+2]);
    double val3 = exp(softmax_input[i+3]);

    // Store results in exp_results array
    exp_results[i] = val0;
    exp_results[i+1] = val1;
    exp_results[i+2] = val2;
    exp_results[i+3] = val3;

    // Accumulate sum for unrolled elements.
    // Summing the unrolled values first before adding to sum_total_0 can
    // potentially reduce dependency chain on sum_total_0 and improve ILP.
    sum_total_0 += (val0 + val1 + val2 + val3);
  }

  // Handle any remaining elements (if size is not a multiple of unroll_factor)
  for (; i < size; ++i) {
    exp_results[i] = exp(softmax_input[i]);
    sum_total_0 += exp_results[i];
  }

  // Optimization 2: Replace division with multiplication by pre-calculating the inverse.
  // Division operations are generally more computationally expensive than multiplications.
  // By calculating 1.0 / sum_total_0 once, we replace 'size' divisions with one division
  // and 'size' multiplications, leading to significant performance improvement.
  double inv_sum_total_0 = 1.0 / sum_total_0;

  // Optimization 3: Loop Unrolling for the second loop.
  // Similar to the first loop, this helps with loop overhead and instruction scheduling,
  // and is highly amenable to SIMD vectorization by the compiler.
  i = 0; // Reset index for the second loop

  // Process elements in chunks of 'unroll_factor'
  for (; i + unroll_factor <= size; i += unroll_factor) {
    softmax_output[i] = exp_results[i] * inv_sum_total_0;
    softmax_output[i+1] = exp_results[i+1] * inv_sum_total_0;
    softmax_output[i+2] = exp_results[i+2] * inv_sum_total_0;
    softmax_output[i+3] = exp_results[i+3] * inv_sum_total_0;
  }

  // Handle any remaining elements
  for (; i < size; ++i) {
    softmax_output[i] = exp_results[i] * inv_sum_total_0;
  }
}