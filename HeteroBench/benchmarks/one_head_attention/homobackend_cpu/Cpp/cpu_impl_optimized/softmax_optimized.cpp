#include "cpu_impl.h"

#include <math.h>   // For exp and INFINITY
#include <algorithm> // For std::max

// Optimized function implementation
#include <cmath>     // For exp and INFINITY
#include <algorithm> // For std::max

// The problem statement implies that necessary headers like <cmath> and <algorithm>
// are already included by the compilation environment or cpu_impl.h.
// The original code uses `max` and `INFINITY` without explicit includes in the provided snippet,
// which suggests they are available in the context.

void softmax_optimized(double *softmax_x, double *softmax_output, 
             int batch_size, int input_h, int input_w, int axis) {
    // Allocate temporary arrays on the heap, as in the original implementation.
    // These arrays are necessary to store intermediate results due to the
    // reduction operations (max and sum) along the innermost dimension.
    double *softmax_m = new double[batch_size * input_h];
    double *softmax_exp_result = new double[batch_size * input_h * input_w];
    double *softmax_l = new double[batch_size * input_h];

    // Optimization 1: Loop Fusion and Strength Reduction for Array Indexing
    // Instead of calling separate functions (get_max, get_exp, get_sum) and having
    // multiple passes over the data, we fuse the operations into fewer, more efficient loops.
    // This significantly improves data locality by keeping relevant data in cache
    // and reduces memory traffic. Strength reduction is applied by precomputing
    // base indices for array access within loops. Using `long long` for indices
    // to prevent potential integer overflow for very large dimensions.

    // Pass 1: Calculate max_x for each (i, j) slice.
    // This replaces the functionality of the `get_max` function.
    // The maximum value for each (i, j) slice is accumulated in `current_max`.
    for (int i = 0; i < batch_size; ++i) {
        // Precompute i-dependent offsets for `softmax_m` and `softmax_x`
        long long i_offset_m = (long long)i * input_h;
        long long i_offset_x = (long long)i * input_h * input_w;

        for (int j = 0; j < input_h; ++j) {
            double current_max = -INFINITY; // Accumulator for max, kept in a CPU register
            
            // Precompute j-dependent offset for `softmax_x` within the current (i, j) slice
            long long current_base_idx_x = i_offset_x + (long long)j * input_w;
            
            // Optimization 2: Loop Unrolling for the innermost k loop (by 4)
            // This exposes more instruction-level parallelism to the CPU and reduces loop overhead.
            // It allows the CPU to fetch and execute multiple instructions concurrently.
            int k = 0;
            for (; k + 3 < input_w; k += 4) {
                current_max = std::max(current_max, softmax_x[current_base_idx_x + k]);
                current_max = std::max(current_max, softmax_x[current_base_idx_x + k + 1]);
                current_max = std::max(current_max, softmax_x[current_base_idx_x + k + 2]);
                current_max = std::max(current_max, softmax_x[current_base_idx_x + k + 3]);
            }
            // Handle remainder elements if input_w is not a multiple of 4
            for (; k < input_w; ++k) {
                current_max = std::max(current_max, softmax_x[current_base_idx_x + k]);
            }
            softmax_m[i_offset_m + j] = current_max;
        }
    }

    // Pass 2: Calculate exp(x - m) and sum(exp_result).
    // This fuses the `x - m` calculation, `get_exp`, and `get_sum` functionalities.
    // `current_sum_exp` accumulates the sum of exponentials for each (i, j) slice.
    for (int i = 0; i < batch_size; ++i) {
        // Precompute i-dependent offsets for `softmax_m`, `softmax_l`, `softmax_x`, and `softmax_exp_result`
        long long i_offset_m = (long long)i * input_h;
        long long i_offset_l = (long long)i * input_h;
        long long i_offset_x = (long long)i * input_h * input_w;
        long long i_offset_exp_result = (long long)i * input_h * input_w;

        for (int j = 0; j < input_h; ++j) {
            double current_sum_exp = 0.0; // Accumulator for sum, kept in a CPU register
            double m_val = softmax_m[i_offset_m + j]; // Load m_val once per (i,j) block, kept in a CPU register

            // Precompute j-dependent offsets for `softmax_x` and `softmax_exp_result`
            long long current_base_idx_x = i_offset_x + (long long)j * input_w;
            long long current_base_idx_exp_result = i_offset_exp_result + (long long)j * input_w;

            // Optimization 2: Loop Unrolling for the innermost k loop (by 4)
            int k = 0;
            for (; k + 3 < input_w; k += 4) {
                // Calculate x - m
                double val0 = softmax_x[current_base_idx_x + k] - m_val;
                double val1 = softmax_x[current_base_idx_x + k + 1] - m_val;
                double val2 = softmax_x[current_base_idx_x + k + 2] - m_val;
                double val3 = softmax_x[current_base_idx_x + k + 3] - m_val;

                // Calculate exp(x - m)
                // `exp` is a transcendental function, its performance is largely dependent
                // on the underlying math library implementation.
                double exp_val0 = exp(val0);
                double exp_val1 = exp(val1);
                double exp_val2 = exp(val2);
                double exp_val3 = exp(val3);

                // Store exp_result for the final division in Pass 3
                softmax_exp_result[current_base_idx_exp_result + k] = exp_val0;
                softmax_exp_result[current_base_idx_exp_result + k + 1] = exp_val1;
                softmax_exp_result[current_base_idx_exp_result + k + 2] = exp_val2;
                softmax_exp_result[current_base_idx_exp_result + k + 3] = exp_val3;

                // Accumulate sum of exponentials
                current_sum_exp += exp_val0 + exp_val1 + exp_val2 + exp_val3;
            }
            // Handle remainder elements
            for (; k < input_w; ++k) {
                double val = softmax_x[current_base_idx_x + k] - m_val;
                double exp_val = exp(val);
                softmax_exp_result[current_base_idx_exp_result + k] = exp_val;
                current_sum_exp += exp_val;
            }
            softmax_l[i_offset_l + j] = current_sum_exp;
        }
    }

    // Pass 3: Calculate exp_result / l.
    // This replaces the final division loop in the original `softmax` function.
    for (int i = 0; i < batch_size; ++i) {
        // Precompute i-dependent offsets for `softmax_l`, `softmax_exp_result`, and `softmax_output`
        long long i_offset_l = (long long)i * input_h;
        long long i_offset_exp_result = (long long)i * input_h * input_w;
        long long i_offset_output = (long long)i * input_h * input_w;

        for (int j = 0; j < input_h; ++j) {
            double l_val = softmax_l[i_offset_l + j]; // Load l_val once per (i,j) block, kept in a CPU register

            // Precompute j-dependent offsets for `softmax_exp_result` and `softmax_output`
            long long current_base_idx_exp_result = i_offset_exp_result + (long long)j * input_w;
            long long current_base_idx_output = i_offset_output + (long long)j * input_w;

            // Optimization 2: Loop Unrolling for the innermost k loop (by 4)
            int k = 0;
            for (; k + 3 < input_w; k += 4) {
                softmax_output[current_base_idx_output + k] = softmax_exp_result[current_base_idx_exp_result + k] / l_val;
                softmax_output[current_base_idx_output + k + 1] = softmax_exp_result[current_base_idx_exp_result + k + 1] / l_val;
                softmax_output[current_base_idx_output + k + 2] = softmax_exp_result[current_base_idx_exp_result + k + 2] / l_val;
                softmax_output[current_base_idx_output + k + 3] = softmax_exp_result[current_base_idx_exp_result + k + 3] / l_val;
            }
            // Handle remainder elements
            for (; k < input_w; ++k) {
                softmax_output[current_base_idx_output + k] = softmax_exp_result[current_base_idx_exp_result + k] / l_val;
            }
        }
    }

    // Deallocate temporary memory to prevent memory leaks.
    delete[] softmax_m;
    delete[] softmax_exp_result;
    delete[] softmax_l;
}