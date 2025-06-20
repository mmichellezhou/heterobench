#include "cpu_impl.h"

void softmax_optimized(double *softmax_input, double *exp_results, double *softmax_output, int size)
{
  // Accumulators for sum_total_0 to enable instruction-level parallelism
  // by breaking the data dependency chain for the sum.
  double sum_total_0_part0 = 0.0;
  double sum_total_0_part1 = 0.0;
  double sum_total_0_part2 = 0.0;
  double sum_total_0_part3 = 0.0;
  double sum_total_0_part4 = 0.0;
  double sum_total_0_part5 = 0.0;
  double sum_total_0_part6 = 0.0;
  double sum_total_0_part7 = 0.0;

  int i = 0;
  // Calculate limit for the main unrolled loop.
  // Using an unroll factor of 8 to reduce loop overhead and expose Instruction-Level Parallelism (ILP)
  // for the expensive `exp` function calls and subsequent additions.
  int limit = size - (size % 8);

  // First loop: calculate exp(softmax_input[i]) and accumulate sum_total_0.
  // Loop unrolling and register optimization are applied here.
  for (; i < limit; i += 8) {
    // Compute exp for 8 elements. These operations can potentially execute in parallel
    // on modern CPUs due to multiple floating-point execution units.
    double val0 = exp(softmax_input[i]);
    double val1 = exp(softmax_input[i+1]);
    double val2 = exp(softmax_input[i+2]);
    double val3 = exp(softmax_input[i+3]);
    double val4 = exp(softmax_input[i+4]);
    double val5 = exp(softmax_input[i+5]);
    double val6 = exp(softmax_input[i+6]);
    double val7 = exp(softmax_input[i+7]);

    // Store results into exp_results array.
    exp_results[i] = val0;
    exp_results[i+1] = val1;
    exp_results[i+2] = val2;
    exp_results[i+3] = val3;
    exp_results[i+4] = val4;
    exp_results[i+5] = val5;
    exp_results[i+6] = val6;
    exp_results[i+7] = val7;

    // Accumulate sums into separate registers. This allows the additions to proceed
    // in parallel, reducing the critical path latency of the sum reduction.
    sum_total_0_part0 += val0;
    sum_total_0_part1 += val1;
    sum_total_0_part2 += val2;
    sum_total_0_part3 += val3;
    sum_total_0_part4 += val4;
    sum_total_0_part5 += val5;
    sum_total_0_part6 += val6;
    sum_total_0_part7 += val7;
  }

  // Combine the partial sums from the unrolled loop.
  double sum_total_0 = sum_total_0_part0 + sum_total_0_part1 + sum_total_0_part2 + sum_total_0_part3 +
                       sum_total_0_part4 + sum_total_0_part5 + sum_total_0_part6 + sum_total_0_part7;

  // Handle any remaining elements that did not fit into the unrolled loop (remainder loop).
  for (; i < size; i++) {
    exp_results[i] = exp(softmax_input[i]);
    sum_total_0 += exp_results[i];
  }

  // Second loop: calculate softmax_output.
  // Apply strength reduction: replace division by a multiplication with the reciprocal.
  // This moves the expensive division operation outside the loop.
  double inv_sum_total_0 = 1.0 / sum_total_0;

  // Reset loop index for the second loop.
  i = 0;
  // Apply loop unrolling by 8 for the division part as well, similar to the first loop.
  for (; i < limit; i += 8) {
    softmax_output[i] = exp_results[i] * inv_sum_total_0;
    softmax_output[i+1] = exp_results[i+1] * inv_sum_total_0;
    softmax_output[i+2] = exp_results[i+2] * inv_sum_total_0;
    softmax_output[i+3] = exp_results[i+3] * inv_sum_total_0;
    softmax_output[i+4] = exp_results[i+4] * inv_sum_total_0;
    softmax_output[i+5] = exp_results[i+5] * inv_sum_total_0;
    softmax_output[i+6] = exp_results[i+6] * inv_sum_total_0;
    softmax_output[i+7] = exp_results[i+7] * inv_sum_total_0;
  }

  // Handle any remaining elements for the second loop.
  for (; i < size; i++) {
    softmax_output[i] = exp_results[i] * inv_sum_total_0;
  }
}