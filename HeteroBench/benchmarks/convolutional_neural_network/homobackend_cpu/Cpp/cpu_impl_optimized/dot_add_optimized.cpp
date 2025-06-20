#include "cpu_impl.h"

#include <algorithm> // Required for std::min

#include <algorithm> // Required for std::min

#include <algorithm> // Required for std::min

void dot_add_optimized(double *dot_add_input_x, double *dot_add_input_W, double *dot_add_input_b, double *dot_add_output, int x_h, int x_w, int W_h, int W_w) {
  // Define local constants for tiling and unrolling factors.
  // These values are chosen as common heuristics for double-precision floating-point operations
  // on modern CPUs, balancing cache locality, instruction-level parallelism, and register pressure.
  // TILE_J: A tile size for the 'j' dimension (columns of W and output).
  //         A value of 64 doubles (512 bytes) is small enough to fit comfortably in L1 cache.
  const int TILE_J = 64;
  // UNROLL_J: Unroll factor for the innermost 'j' loop.
  //           Unrolling by 4 exposes more independent operations for the CPU's execution units,
  //           reducing loop overhead and improving instruction-level parallelism (ILP).
  const int UNROLL_J = 4;

  // Phase 1: Initialize the output matrix with the bias values.
  // This combines the initialization step and the bias addition, eliminating a separate loop nest later.
  // The matrix multiplication will then accumulate its results on top of these initial bias values.
  for (int i = 0; i < x_h; i++) {
    // Use pointer arithmetic for strength reduction, avoiding repeated multiplications.
    double* current_output_row_ptr = dot_add_output + i * W_w;
    // Unroll the inner loop for better performance during initialization.
    for (int j = 0; j < W_w; j += UNROLL_J) {
      // Handle tail cases where W_w is not a multiple of UNROLL_J.
      if (j + 0 < W_w) {
        current_output_row_ptr[j + 0] = dot_add_input_b[j + 0];
      }
      if (j + 1 < W_w) {
        current_output_row_ptr[j + 1] = dot_add_input_b[j + 1];
      }
      if (j + 2 < W_w) {
        current_output_row_ptr[j + 2] = dot_add_input_b[j + 2];
      }
      if (j + 3 < W_w) {
        current_output_row_ptr[j + 3] = dot_add_input_b[j + 3];
      }
    }
  }

  // Phase 2: Perform the matrix multiplication (X * W) and accumulate results onto the bias-initialized output.
  // The original loop order was ijk. This is transformed to ikj for improved cache performance.
  // In the ikj order:
  // - Accesses to dot_add_input_x[i * x_w + k] are sequential for 'k' within an 'i' row.
  // - Accesses to dot_add_input_W[k * W_w + j] are sequential for 'j' within a 'k' row.
  // - Accesses to dot_add_output[i * W_w + j] are sequential for 'j' within an 'i' row.
  // This ensures stride-1 memory access patterns for the most frequently accessed data in the innermost loop,
  // maximizing cache line utilization and reducing cache misses.
  for (int i = 0; i < x_h; i++) {
    // Pointer to the current row of X (dot_add_input_x[i][:])
    double* current_x_row_ptr = dot_add_input_x + i * x_w;
    // Pointer to the current row of Output (dot_add_output[i][:])
    double* current_output_row_ptr = dot_add_output + i * W_w;

    for (int k = 0; k < x_w; k++) {
      // Load x_val once per (i, k) pair. This value is then reused W_w times in the inner 'j' loop,
      // promoting register reuse and reducing memory loads.
      double x_val = current_x_row_ptr[k];
      // Pointer to the current row of W (dot_add_input_W[k][:])
      double* current_W_row_ptr = dot_add_input_W + k * W_w;

      // Tiled loop for 'j' dimension.
      // This helps keep a smaller, more cache-friendly block of 'W' and 'Output' data in L1/L2 cache,
      // especially when W_w is very large.
      for (int jj = 0; jj < W_w; jj += TILE_J) {
        // Calculate the actual end index for the current 'j' tile, handling boundary conditions.
        int j_end_tile = std::min(jj + TILE_J, W_w);

        // Unrolled loop for 'j' dimension.
        // This reduces loop control overhead (branching, incrementing 'j') and exposes more
        // independent multiply-add operations to the CPU's execution units, improving ILP.
        for (int j = jj; j < j_end_tile; j += UNROLL_J) {
          // Perform UNROLL_J multiply-add operations.
          // Each operation is independent, allowing the CPU to pipeline them effectively.
          // Explicitly check bounds for tail cases within the unrolled block to ensure correctness.
          if (j + 0 < j_end_tile) {
            current_output_row_ptr[j + 0] += x_val * current_W_row_ptr[j + 0];
          }
          if (j + 1 < j_end_tile) {
            current_output_row_ptr[j + 1] += x_val * current_W_row_ptr[j + 1];
          }
          if (j + 2 < j_end_tile) {
            current_output_row_ptr[j + 2] += x_val * current_W_row_ptr[j + 2];
          }
          if (j + 3 < j_end_tile) {
            current_output_row_ptr[j + 3] += x_val * current_W_row_ptr[j + 3];
          }
        }
      }
    }
  }
}