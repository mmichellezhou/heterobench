#include "cpu_impl.h"

using namespace std;

void kernel_3mm_2_optimized(double E[NI + 0][NJ + 0], double F[NJ + 0][NL + 0], double G[NI + 0][NL + 0])
{
  // Define block size for tiling and unroll factor for the innermost loop.
  // These are declared as const int within the function scope, satisfying the constraint
  // that no constants, macros, or types need to be defined outside the function body.
  const int BS = 128; // Block size for cache tiling. A common size that allows blocks
                      // of E, F, and G to fit in L1/L2 cache for double precision.
  const int UNROLL_C6 = 4; // Unroll factor for the innermost loop (c6).
                           // This helps expose instruction-level parallelism (ILP)
                           // by allowing the CPU to pipeline multiple independent
                           // multiply-add operations.

  int c1, c2, c6;
  int c1_tile, c2_tile, c6_tile;

  // Loop Tiling: Iterate over blocks of the matrices. This improves cache locality
  // by ensuring that data accessed within a block stays in cache for longer,
  // reducing memory access latency.
  for (c1_tile = 0; c1_tile < NI; c1_tile += BS) {
    for (c2_tile = 0; c2_tile < NJ; c2_tile += BS) {
      for (c6_tile = 0; c6_tile < NL; c6_tile += BS) {
        // Inner loops for the current tile.
        // The loop bounds ensure we don't go out of bounds for the matrices
        // and process only within the current tile.
        for (c1 = c1_tile; c1 < NI && c1 < c1_tile + BS; c1++) {
          for (c2 = c2_tile; c2 < NJ && c2 < c2_tile + BS; c2++) {
            // Register Optimization: Load E[c1][c2] once into a local variable (register)
            // before the innermost loop. This avoids redundant memory loads of E[c1][c2]
            // within the c6 loop, as its value is constant for a given c1 and c2.
            double E_val = E[c1][c2];

            // Determine the actual upper bound for the innermost c6 loop within the tile.
            int c6_limit_for_tile = c6_tile + BS;
            if (c6_limit_for_tile > NL) {
                c6_limit_for_tile = NL;
            }

            // Unrolling: Process multiple iterations of the innermost loop (c6) at once.
            // This reduces loop overhead and allows the CPU to find more independent
            // instructions to execute in parallel (Instruction-Level Parallelism).
            // Calculate the limit for the unrolled part of the loop.
            int c6_unrolled_limit = c6_limit_for_tile - (c6_limit_for_tile - c6_tile) % UNROLL_C6;

            for (c6 = c6_tile; c6 < c6_unrolled_limit; c6 += UNROLL_C6) {
              // Each line below represents an independent multiply-add operation.
              // The compiler can schedule these operations to utilize multiple
              // execution units on the CPU.
              G[c1][c6]     += E_val * F[c2][c6];
              G[c1][c6 + 1] += E_val * F[c2][c6 + 1];
              G[c1][c6 + 2] += E_val * F[c2][c6 + 2];
              G[c1][c6 + 3] += E_val * F[c2][c6 + 3];
            }

            // Remainder Loop: Handle any remaining iterations that couldn't be processed
            // by the unrolled loop (i.e., when the total iterations are not a multiple of UNROLL_C6).
            for (; c6 < c6_limit_for_tile; c6++) {
              G[c1][c6] += E_val * F[c2][c6];
            }
          }
        }
      }
    }
  }
}
