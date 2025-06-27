#include "cpu_impl.h"

using namespace std;

void kernel_3mm_0_optimized(double A[NI + 0][NK + 0], double B[NK + 0][NJ + 0], double E[NI + 0][NJ + 0])
{
  // Define block sizes and unroll factor locally within the function scope.
  // These are chosen to optimize cache utilization and instruction-level parallelism.
  // For a double (8 bytes), a 64x64 block is 32KB. Three such blocks (for A, B, E)
  // would be approximately 96KB, fitting comfortably within the 1.5MB L1d cache.
  const int BLOCK_SIZE = 64;

  // Unroll factor for the innermost loop.
  // With 16 available scalar double registers on Intel Xeon, an unroll factor of 4
  // allows for 1 A_val, 4 E_acc, and 4 B_val registers (total 9), which is efficient.
  const int UNROLL_FACTOR = 4;

  int c1_block, c2_block, c5_block;
  int c1, c2, c5;

  // Tiled loops for cache locality.
  // The order of block loops (c1_block, c2_block, c5_block) is chosen to
  // facilitate accumulation into E blocks.
  for (c1_block = 0; c1_block < NI; c1_block += BLOCK_SIZE) {
    for (c2_block = 0; c2_block < NJ; c2_block += BLOCK_SIZE) {
      for (c5_block = 0; c5_block < NK; c5_block += BLOCK_SIZE) {

        // Inner loops operating on the current blocks.
        // Loop order (c1, c5, c2) is chosen to ensure stride-1 memory access
        // for B and E in the innermost loop, significantly improving cache performance.
        for (c1 = c1_block; c1 < (c1_block + BLOCK_SIZE < NI ? c1_block + BLOCK_SIZE : NI); c1++) {
          for (c5 = c5_block; c5 < (c5_block + BLOCK_SIZE < NK ? c5_block + BLOCK_SIZE : NK); c5++) {

            // Register optimization: Load A[c1][c5] once per c5 iteration.
            // This value is invariant within the innermost c2 loop.
            double A_val = A[c1][c5];

            // Calculate the effective end of the current c2 block.
            int c2_loop_end = (c2_block + BLOCK_SIZE < NJ ? c2_block + BLOCK_SIZE : NJ);
            // Calculate the upper bound for the main unrolled loop to handle tail iterations.
            int c2_unrolled_end = c2_loop_end - ((c2_loop_end - c2_block) % UNROLL_FACTOR);

            // Unrolled loop for c2 to expose instruction-level parallelism and reduce loop overhead.
            for (c2 = c2_block; c2 < c2_unrolled_end; c2 += UNROLL_FACTOR) {
              // Scalar register blocking: Load E[c1][c2] values into registers,
              // accumulate, and write back only once per unrolled block.
              double E_acc0 = E[c1][c2];
              double E_acc1 = E[c1][c2 + 1];
              double E_acc2 = E[c1][c2 + 2];
              double E_acc3 = E[c1][c2 + 3];

              // Perform the multiply-add operations.
              // The compiler can potentially fuse these into FMA instructions.
              E_acc0 += A_val * B[c5][c2];
              E_acc1 += A_val * B[c5][c2 + 1];
              E_acc2 += A_val * B[c5][c2 + 2];
              E_acc3 += A_val * B[c5][c2 + 3];

              // Write back the accumulated values to memory.
              E[c1][c2]     = E_acc0;
              E[c1][c2 + 1] = E_acc1;
              E[c1][c2 + 2] = E_acc2;
              E[c1][c2 + 3] = E_acc3;
            }

            // Tail loop for remaining c2 iterations that are not a multiple of UNROLL_FACTOR.
            for (c2 = c2_unrolled_end; c2 < c2_loop_end; c2++) {
              E[c1][c2] += A_val * B[c5][c2];
            }
          }
        }
      }
    }
  }
}
