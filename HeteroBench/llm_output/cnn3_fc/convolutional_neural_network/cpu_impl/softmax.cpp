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

// Using 'restrict' keyword to inform the compiler that pointers do not alias.
// This allows for more aggressive optimizations, including better auto-vectorization.
// For optimal performance, compile with aggressive optimization flags such as:
// -O3 -march=native -ffast-math (for GCC/Clang)
// Note: -ffast-math can reorder floating-point operations and might affect
// numerical precision in some cases, but often provides significant speedups
// for mathematical functions like exp and division.
void softmax(double *restrict softmax_input, double *restrict exp_results, double *restrict softmax_output, int size) 
{
  // Use multiple accumulators for the sum reduction. This technique helps
  // break the data dependency chain on 'sum_total_0', allowing the CPU to
  // execute additions in parallel (instruction-level parallelism) and
  // potentially enabling better auto-vectorization by the compiler.
  // An unroll factor of 8 is chosen as it aligns well with common SIMD
  // register sizes (e.g., 8 doubles for AVX-512, or 4 doubles for AVX/AVX2
  // where unrolling by 8 can still improve instruction scheduling).
  double sum_total_0_0 = 0.0;
  double sum_total_0_1 = 0.0;
  double sum_total_0_2 = 0.0;
  double sum_total_0_3 = 0.0;
  double sum_total_0_4 = 0.0;
  double sum_total_0_5 = 0.0;
  double sum_total_0_6 = 0.0;
  double sum_total_0_7 = 0.0;

  int i = 0;
  const int unroll_factor = 8;

  // Loop 1: Calculate exp(input) and accumulate sum
  // Unrolled loop for the main part of the array
  for (; i + (unroll_factor - 1) < size; i += unroll_factor) {
    exp_results[i] = exp(softmax_input[i]);
    sum_total_0_0 += exp_results[i];

    exp_results[i+1] = exp(softmax_input[i+1]);
    sum_total_0_1 += exp_results[i+1];

    exp_results[i+2] = exp(softmax_input[i+2]);
    sum_total_0_2 += exp_results[i+2];

    exp_results[i+3] = exp(softmax_input[i+3]);
    sum_total_0_3 += exp_results[i+3];

    exp_results[i+4] = exp(softmax_input[i+4]);
    sum_total_0_4 += exp_results[i+4];

    exp_results[i+5] = exp(softmax_input[i+5]);
    sum_total_0_5 += exp_results[i+5];

    exp_results[i+6] = exp(softmax_input[i+6]);
    sum_total_0_6 += exp_results[i+6];

    exp_results[i+7] = exp(softmax_input[i+7]);
    sum_total_0_7 += exp_results[i+7];
  }

  // Combine the partial sums from the unrolled loop
  double sum_total_0 = sum_total_0_0 + sum_total_0_1 + sum_total_0_2 + sum_total_0_3 +
                       sum_total_0_4 + sum_total_0_5 + sum_total_0_6 + sum_total_0_7;

  // Scalar cleanup loop for any remaining elements (if size is not a multiple of unroll_factor)
  for (; i < size; i++) {
    exp_results[i] = exp(softmax_input[i]);
    sum_total_0 += exp_results[i];
  }

  // Loop 2: Calculate softmax_output by dividing by the total sum
  // Reset index for the second loop
  i = 0; 
  // Unrolled loop for the main part of the array
  for (; i + (unroll_factor - 1) < size; i += unroll_factor) {
    softmax_output[i] = exp_results[i] / sum_total_0;
    softmax_output[i+1] = exp_results[i+1] / sum_total_0;
    softmax_output[i+2] = exp_results[i+2] / sum_total_0;
    softmax_output[i+3] = exp_results[i+3] / sum_total_0;
    softmax_output[i+4] = exp_results[i+4] / sum_total_0;
    softmax_output[i+5] = exp_results[i+5] / sum_total_0;
    softmax_output[i+6] = exp_results[i+6] / sum_total_0;
    softmax_output[i+7] = exp_results[i+7] / sum_total_0;
  }

  // Scalar cleanup loop for any remaining elements
  for (; i < size; i++) {
    softmax_output[i] = exp_results[i] / sum_total_0;
  }
}