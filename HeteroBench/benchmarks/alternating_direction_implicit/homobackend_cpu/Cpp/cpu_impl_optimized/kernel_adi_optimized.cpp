#include "cpu_impl.h"

void kernel_adi_optimized(int tsteps,int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  // Constants for unrolling and tiling.
  // These values are chosen to balance register pressure, instruction-level parallelism (ILP),
  // and cache locality for typical CPU architectures (e.g., Intel Xeon Gold 6248R).
  // For double-precision floating-point numbers (8 bytes), a cache line (64 bytes) holds 8 doubles.
  // U_H: Unroll factor for horizontal sweeps (inner c8 loop).
  // BLOCK_C2: Tiling factor for the c2 loop in vertical sweeps to improve cache locality.
  //           A block size of 16 means 16 doubles (128 bytes) are processed contiguously
  //           in the innermost loop, which should fit well within L1 cache and exploit spatial locality.
  // U_V: Unroll factor for the innermost c2 loop within the tiled vertical sweeps.
  const int U_H = 4;
  const int BLOCK_C2 = 16;
  const int U_V = 4;

  // Main time-stepping loop
  for (int c0 = 0; c0 <= TSTEPS; ++c0) {
    // Phase 1: Horizontal sweeps (row-major access, inherently good cache locality)
    for (int c2 = 0; c2 < N; ++c2) { // Iterate over rows
      // Fused forward sweeps for B and X
      // Original:
      // for (c8 = 1; c8 <= N - 1; c8++) { B[c2][c8] = B[c2][c8] - A[c2][c8] * A[c2][c8] / B[c2][c8 - 1]; }
      // for (c8 = 1; c8 <= N - 1; c8++) { X[c2][c8] = X[c2][c8] - X[c2][c8 - 1] * A[c2][c8] / B[c2][c8 - 1]; }
      // Optimizations: Loop fusion, strength reduction (division to multiplication by reciprocal), loop unrolling.
      int c8_limit_unrolled = N - 1 - ((N - 1 - 1 + 1) % U_H); // Calculate upper bound for unrolled loop
      for (int c8 = 1; c8 <= c8_limit_unrolled; c8 += U_H) {
        // Iteration c8
        double inv_B_prev_0 = 1.0 / B[c2][c8 - 1];
        double common_factor_0 = A[c2][c8] * inv_B_prev_0;
        B[c2][c8] = B[c2][c8] - A[c2][c8] * common_factor_0;
        X[c2][c8] = X[c2][c8] - X[c2][c8 - 1] * common_factor_0;

        // Iteration c8 + 1
        double inv_B_prev_1 = 1.0 / B[c2][c8]; // B[c2][c8] is the newly computed value
        double common_factor_1 = A[c2][c8 + 1] * inv_B_prev_1;
        B[c2][c8 + 1] = B[c2][c8 + 1] - A[c2][c8 + 1] * common_factor_1;
        X[c2][c8 + 1] = X[c2][c8 + 1] - X[c2][c8] * common_factor_1;

        // Iteration c8 + 2
        double inv_B_prev_2 = 1.0 / B[c2][c8 + 1];
        double common_factor_2 = A[c2][c8 + 2] * inv_B_prev_2;
        B[c2][c8 + 2] = B[c2][c8 + 2] - A[c2][c8 + 2] * common_factor_2;
        X[c2][c8 + 2] = X[c2][c8 + 2] - X[c2][c8 + 1] * common_factor_2;

        // Iteration c8 + 3
        double inv_B_prev_3 = 1.0 / B[c2][c8 + 2];
        double common_factor_3 = A[c2][c8 + 3] * inv_B_prev_3;
        B[c2][c8 + 3] = B[c2][c8 + 3] - A[c2][c8 + 3] * common_factor_3;
        X[c2][c8 + 3] = X[c2][c8 + 3] - X[c2][c8 + 2] * common_factor_3;
      }
      // Remainder loop for forward sweeps
      for (int c8 = c8_limit_unrolled + 1; c8 < N; ++c8) {
        double inv_B_prev = 1.0 / B[c2][c8 - 1];
        double common_factor = A[c2][c8] * inv_B_prev;
        B[c2][c8] = B[c2][c8] - A[c2][c8] * common_factor;
        X[c2][c8] = X[c2][c8] - X[c2][c8 - 1] * common_factor;
      }

      // Backward sweep for X
      // Original:
      // for (c8 = 0; c8 <= N - 3; c8++) { X[c2][N - c8 - 2] = (X[c2][N - 2 - c8] - X[c2][N - 2 - c8 - 1] * A[c2][N - c8 - 3]) / B[c2][N - 3 - c8]; }
      // Transformation: Let k = N - c8 - 2. Loop becomes for (k = N-2; k >= 1; k--)
      // X[c2][k] = (X[c2][k] - X[c2][k - 1] * A[c2][k - 1]) / B[c2][k - 1];
      // Optimizations: Strength reduction (division to multiplication by reciprocal), loop unrolling.
      int k_start = N - 2;
      int k_end_unrolled = 1 + U_H - 1; // Smallest k for unrolled loop
      for (int k = k_start; k >= k_end_unrolled; k -= U_H) {
        // Iteration k
        double inv_B_k_minus_1_0 = 1.0 / B[c2][k - 1];
        X[c2][k] = (X[c2][k] - X[c2][k - 1] * A[c2][k - 1]) * inv_B_k_minus_1_0;

        // Iteration k - 1
        double inv_B_k_minus_1_1 = 1.0 / B[c2][k - 2];
        X[c2][k - 1] = (X[c2][k - 1] - X[c2][k - 2] * A[c2][k - 2]) * inv_B_k_minus_1_1;

        // Iteration k - 2
        double inv_B_k_minus_1_2 = 1.0 / B[c2][k - 3];
        X[c2][k - 2] = (X[c2][k - 2] - X[c2][k - 3] * A[c2][k - 3]) * inv_B_k_minus_1_2;

        // Iteration k - 3
        double inv_B_k_minus_1_3 = 1.0 / B[c2][k - 4];
        X[c2][k - 3] = (X[c2][k - 3] - X[c2][k - 4] * A[c2][k - 4]) * inv_B_k_minus_1_3;
      }
      // Remainder loop for backward sweep
      for (int k = k_end_unrolled - 1; k >= 1; --k) {
        double inv_B_k_minus_1 = 1.0 / B[c2][k - 1];
        X[c2][k] = (X[c2][k] - X[c2][k - 1] * A[c2][k - 1]) * inv_B_k_minus_1;
      }
    }
    // Final single element division for horizontal sweeps
    // Original: for (c2 = 0; c2 <= N - 1; c2++) { X[c2][N - 1] = X[c2][N - 1] / B[c2][N - 1]; }
    for (int c2 = 0; c2 < N; ++c2) {
      X[c2][N - 1] = X[c2][N - 1] / B[c2][N - 1];
    }

    // Phase 2: Vertical sweeps (column-major access, poor cache locality without tiling)
    // Optimizations: Loop reordering (tiling c2), loop fusion, strength reduction, unrolling c2.
    for (int c2_block = 0; c2_block < N; c2_block += BLOCK_C2) { // Tile c2 (columns)
      // Fused forward sweeps for B and X
      // Original:
      // for (c8 = 1; c8 <= N - 1; c8++) { B[c8][c2] = B[c8][c2] - A[c8][c2] * A[c8][c2] / B[c8 - 1][c2]; }
      // for (c8 = 1; c8 <= N - 1; c8++) { X[c8][c2] = X[c8][c2] - X[c8 - 1][c2] * A[c8][c2] / B[c8 - 1][c2]; }
      for (int c8 = 1; c8 < N; ++c8) { // Iterate over rows (middle loop for dependencies)
        for (int c2 = c2_block; c2 < c2_block + BLOCK_C2 && c2 < N; c2 += U_V) { // Iterate over columns within block (innermost for locality)
          // Iteration c2
          double inv_B_prev_0 = 1.0 / B[c8 - 1][c2];
          double common_factor_0 = A[c8][c2] * inv_B_prev_0;
          B[c8][c2] = B[c8][c2] - A[c8][c2] * common_factor_0;
          X[c8][c2] = X[c8][c2] - X[c8 - 1][c2] * common_factor_0;

          // Iteration c2 + 1
          if (c2 + 1 < c2_block + BLOCK_C2 && c2 + 1 < N) {
            double inv_B_prev_1 = 1.0 / B[c8 - 1][c2 + 1];
            double common_factor_1 = A[c8][c2 + 1] * inv_B_prev_1;
            B[c8][c2 + 1] = B[c8][c2 + 1] - A[c8][c2 + 1] * common_factor_1;
            X[c8][c2 + 1] = X[c8][c2 + 1] - X[c8 - 1][c2 + 1] * common_factor_1;
          }

          // Iteration c2 + 2
          if (c2 + 2 < c2_block + BLOCK_C2 && c2 + 2 < N) {
            double inv_B_prev_2 = 1.0 / B[c8 - 1][c2 + 2];
            double common_factor_2 = A[c8][c2 + 2] * inv_B_prev_2;
            B[c8][c2 + 2] = B[c8][c2 + 2] - A[c8][c2 + 2] * common_factor_2;
            X[c8][c2 + 2] = X[c8][c2 + 2] - X[c8 - 1][c2 + 2] * common_factor_2;
          }

          // Iteration c2 + 3
          if (c2 + 3 < c2_block + BLOCK_C2 && c2 + 3 < N) {
            double inv_B_prev_3 = 1.0 / B[c8 - 1][c2 + 3];
            double common_factor_3 = A[c8][c2 + 3] * inv_B_prev_3;
            B[c8][c2 + 3] = B[c8][c2 + 3] - A[c8][c2 + 3] * common_factor_3;
            X[c8][c2 + 3] = X[c8][c2 + 3] - X[c8 - 1][c2 + 3] * common_factor_3;
          }
        }
      }

      // Backward sweep for X
      // Original:
      // for (c8 = 0; c8 <= N - 3; c8++) { X[N - 2 - c8][c2] = (X[N - 2 - c8][c2] - X[N - c8 - 3][c2] * A[N - 3 - c8][c2]) / B[N - 2 - c8][c2]; }
      // Transformation: Let k = N - 2 - c8. Loop becomes for (k = N-2; k >= 1; k--)
      // X[k][c2] = (X[k][c2] - X[k - 1][c2] * A[k - 1][c2]) / B[k][c2];
      // Optimizations: Strength reduction (division to multiplication by reciprocal), unrolling c2.
      for (int k = N - 2; k >= 1; --k) { // Iterate over rows (outer loop for dependencies)
        for (int c2 = c2_block; c2 < c2_block + BLOCK_C2 && c2 < N; c2 += U_V) { // Iterate over columns within block (innermost for locality)
          // Iteration c2
          double inv_B_k_0 = 1.0 / B[k][c2];
          X[k][c2] = (X[k][c2] - X[k - 1][c2] * A[k - 1][c2]) * inv_B_k_0;

          // Iteration c2 + 1
          if (c2 + 1 < c2_block + BLOCK_C2 && c2 + 1 < N) {
            double inv_B_k_1 = 1.0 / B[k][c2 + 1];
            X[k][c2 + 1] = (X[k][c2 + 1] - X[k - 1][c2 + 1] * A[k - 1][c2 + 1]) * inv_B_k_1;
          }

          // Iteration c2 + 2
          if (c2 + 2 < c2_block + BLOCK_C2 && c2 + 2 < N) {
            double inv_B_k_2 = 1.0 / B[k][c2 + 2];
            X[k][c2 + 2] = (X[k][c2 + 2] - X[k - 1][c2 + 2] * A[k - 1][c2 + 2]) * inv_B_k_2;
          }

          // Iteration c2 + 3
          if (c2 + 3 < c2_block + BLOCK_C2 && c2 + 3 < N) {
            double inv_B_k_3 = 1.0 / B[k][c2 + 3];
            X[k][c2 + 3] = (X[k][c2 + 3] - X[k - 1][c2 + 3] * A[k - 1][c2 + 3]) * inv_B_k_3;
          }
        }
      }
    }
    // Final single element division for vertical sweeps
    // Original: for (c2 = 0; c2 <= N - 1; c2++) { X[N - 1][c2] = X[N - 1][c2] / B[N - 1][c2]; }
    for (int c2 = 0; c2 < N; ++c2) {
      X[N - 1][c2] = X[N - 1][c2] / B[N - 1][c2];
    }
  }
}