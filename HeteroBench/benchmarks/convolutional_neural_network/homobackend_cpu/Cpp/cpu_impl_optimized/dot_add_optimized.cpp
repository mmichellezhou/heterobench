#include "cpu_impl.h"

#include <algorithm> // Required for std::min

#include <algorithm> // Required for std::min

#include <algorithm> // Required for std::min

void dot_add_optimized(double *dot_add_input_x, double *dot_add_input_W, double *dot_add_input_b, double *dot_add_output, int x_h, int x_w, int W_h, int W_w) {
  // Define tile sizes locally. These are chosen to fit within L1/L2 cache for typical matrix sizes.
  // For Intel Xeon Gold 6248R, L1d is 1.5MB, L2 is 48MB.
  // A tile of 64x64x64 doubles is 3 * 64*64 * 8 bytes = 96KB, which fits well in L1d.
  const int TILE_I = 64; // Rows of X / Output
  const int TILE_J = 64; // Columns of W / Output
  const int TILE_K = 64; // Inner dimension (x_w / W_h)

  // Loop Fusion: Initialize dot_add_output with bias (b)
  // This combines the bias addition step with the initialization of the output matrix,
  // reducing memory writes and improving cache locality for dot_add_output.
  for (int ii = 0; ii < x_h; ii += TILE_I) {
    for (int jj = 0; jj < W_w; jj += TILE_J) {
      // Iterate over the current tile block for rows (i)
      for (int i = ii; i < (ii + TILE_I < x_h ? ii + TILE_I : x_h); i++) {
        double* current_output_row_ptr = dot_add_output + i * W_w;
        int j_limit = (jj + TILE_J < W_w ? jj + TILE_J : W_w);
        // Calculate the limit for the unrolled loop part (unroll factor of 4)
        int j_unrolled_limit = j_limit - (j_limit - jj) % 4;

        // Unroll the innermost loop for better instruction-level parallelism and reduced loop overhead
        for (int j = jj; j < j_unrolled_limit; j += 4) {
          current_output_row_ptr[j]     = dot_add_input_b[j];
          current_output_row_ptr[j + 1] = dot_add_input_b[j + 1];
          current_output_row_ptr[j + 2] = dot_add_input_b[j + 2];
          current_output_row_ptr[j + 3] = dot_add_input_b[j + 3];
        }
        // Handle remaining elements if the loop count is not a multiple of 4
        for (int j = j_unrolled_limit; j < j_limit; j++) {
          current_output_row_ptr[j] = dot_add_input_b[j];
        }
      }
    }
  }

  // Matrix Multiplication (X * W) with optimized loop order and tiling
  // The loop order is transformed from ijk to ikj to improve memory access patterns.
  // Specifically, dot_add_input_W (matrix W) is accessed row-wise (stride 1) in the innermost loop,
  // which is highly cache-friendly.
  // Results are accumulated directly into dot_add_output, which was initialized with bias.
  for (int ii = 0; ii < x_h; ii += TILE_I) { // Outer loop for row blocks of X/Output
    for (int kk = 0; kk < x_w; kk += TILE_K) { // Middle loop for inner dimension blocks (k)
      for (int jj = 0; jj < W_w; jj += TILE_J) { // Inner loop for column blocks of W/Output
        // Iterate over the current tile block for rows (i)
        for (int i = ii; i < (ii + TILE_I < x_h ? ii + TILE_I : x_h); i++) {
          // Use pointer arithmetic to pre-calculate row starting addresses, reducing multiplications inside loops
          double* current_x_row_ptr = dot_add_input_x + i * x_w;
          double* current_output_row_ptr = dot_add_output + i * W_w;

          // Iterate over the current tile block for inner dimension (k)
          for (int k = kk; k < (kk + TILE_K < x_w ? kk + TILE_K : x_w); k++) {
            // Load x[i][k] once per k-loop and keep it in a register (val_x_ik) for reuse
            double val_x_ik = current_x_row_ptr[k];

            // Use pointer arithmetic for W's row starting address
            double* current_W_row_ptr = dot_add_input_W + k * W_w;

            int j_limit = (jj + TILE_J < W_w ? jj + TILE_J : W_w);
            // Calculate the limit for the unrolled loop part (unroll factor of 4)
            int j_unrolled_limit = j_limit - (j_limit - jj) % 4;

            // Unroll the innermost loop (j) for better instruction-level parallelism and reduced loop overhead
            for (int j = jj; j < j_unrolled_limit; j += 4) {
              current_output_row_ptr[j]     += val_x_ik * current_W_row_ptr[j];
              current_output_row_ptr[j + 1] += val_x_ik * current_W_row_ptr[j + 1];
              current_output_row_ptr[j + 2] += val_x_ik * current_W_row_ptr[j + 2];
              current_output_row_ptr[j + 3] += val_x_ik * current_W_row_ptr[j + 3];
            }
            // Handle remaining elements if the loop count is not a multiple of 4
            for (int j = j_unrolled_limit; j < j_limit; j++) {
              current_output_row_ptr[j] += val_x_ik * current_W_row_ptr[j];
            }
          }
        }
      }
    }
  }
}