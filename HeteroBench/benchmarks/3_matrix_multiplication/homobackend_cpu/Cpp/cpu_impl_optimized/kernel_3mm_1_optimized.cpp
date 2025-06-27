#include "cpu_impl.h"

using namespace std;

void kernel_3mm_1_optimized(double C[NJ + 0][NM + 0], double D[NM + 0][NL + 0], double F[NJ + 0][NL + 0])
{
  // Define local constants for tiling and register blocking.
  // TILE_SIZE is chosen to ensure blocks of C, D, and F fit within L1d cache.
  // For double (8 bytes), a 128x128 block is 128KB. Three such blocks (C, D, F)
  // would be 384KB, which fits comfortably within the 1.5MB L1d cache.
  const int TILE_SIZE = 128; 
  
  // UNROLL_I and UNROLL_J define the dimensions of the register block for F.
  // A 4x4 block means 16 double accumulators, which fits well within CPU registers.
  // This helps maximize register reuse and expose instruction-level parallelism.
  const int UNROLL_I = 4; 
  const int UNROLL_J = 4; 

  int c1_outer, c2_outer, c5_outer;
  int c1, c2, c5;

  // Outer loops for tiling (blocking) the matrices.
  // The loop order (c1_outer, c2_outer, c5_outer) corresponds to an 'ijk' block order.
  // This order is chosen to keep blocks of F, C, and D in cache for reuse.
  for (c1_outer = 0; c1_outer < NJ; c1_outer += TILE_SIZE) {
    for (c2_outer = 0; c2_outer < NL; c2_outer += TILE_SIZE) {
      for (c5_outer = 0; c5_outer < NM; c5_outer += TILE_SIZE) {

        // Calculate the actual end bounds for the current tile to handle matrix edges.
        // Using ternary operator as std::min is not allowed per constraints.
        int c1_end_tile = (c1_outer + TILE_SIZE < NJ ? c1_outer + TILE_SIZE : NJ);
        int c2_end_tile = (c2_outer + TILE_SIZE < NL ? c2_outer + TILE_SIZE : NL);
        int c5_end_tile = (c5_outer + TILE_SIZE < NM ? c5_outer + TILE_SIZE : NM);

        // Inner loops for processing the current tile.
        // The loop order (c1, c2, c5) is 'ijk' within the tile.
        // This order allows for register blocking of F[c1][c2] and good cache locality for C[c1][c5].
        // D[c5][c2] access is column-major, but the small tile size helps keep the relevant D block in cache.
        for (c1 = c1_outer; c1 < c1_end_tile; c1 += UNROLL_I) {
          for (c2 = c2_outer; c2 < c2_end_tile; c2 += UNROLL_J) {

            // Declare a local array to hold a block of F values in registers.
            // This is the register blocking optimization.
            double f_acc[UNROLL_I][UNROLL_J];

            // Load the initial values of the F block into registers.
            // Boundary checks are performed to ensure we only access valid elements.
            for (int i_unroll = 0; i_unroll < UNROLL_I; ++i_unroll) {
              for (int j_unroll = 0; j_unroll < UNROLL_J; ++j_unroll) {
                if (c1 + i_unroll < c1_end_tile && c2 + j_unroll < c2_end_tile) {
                  f_acc[i_unroll][j_unroll] = F[c1 + i_unroll][c2 + j_unroll];
                } else {
                  // If out of bounds for F, initialize with 0.0. These values will not be stored back.
                  f_acc[i_unroll][j_unroll] = 0.0; 
                }
              }
            }

            // Innermost loop for accumulation (c5).
            // This loop iterates through the 'k' dimension of the matrix multiplication.
            for (c5 = c5_outer; c5 < c5_end_tile; c5++) {
              // Declare local arrays to hold C and D values for the current c5 and block.
              // This reduces redundant memory accesses within the innermost accumulation loop.
              double c_vals[UNROLL_I];
              double d_vals[UNROLL_J];

              // Load C values for the current c5 and the UNROLL_I rows.
              for (int i_unroll = 0; i_unroll < UNROLL_I; ++i_unroll) {
                if (c1 + i_unroll < c1_end_tile) {
                  c_vals[i_unroll] = C[c1 + i_unroll][c5];
                } else {
                  c_vals[i_unroll] = 0.0; // If out of bounds, set to 0.0 to avoid affecting accumulation.
                }
              }
              // Load D values for the current c5 and the UNROLL_J columns.
              for (int j_unroll = 0; j_unroll < UNROLL_J; ++j_unroll) {
                if (c2 + j_unroll < c2_end_tile) {
                  d_vals[j_unroll] = D[c5][c2 + j_unroll];
                } else {
                  d_vals[j_unroll] = 0.0; // If out of bounds, set to 0.0.
                }
              }

              // Perform the UNROLL_I x UNROLL_J matrix multiplication accumulation.
              // This loop exposes instruction-level parallelism for the FMA operations.
              for (int i_unroll = 0; i_unroll < UNROLL_I; ++i_unroll) {
                for (int j_unroll = 0; j_unroll < UNROLL_J; ++j_unroll) {
                  f_acc[i_unroll][j_unroll] += c_vals[i_unroll] * d_vals[j_unroll];
                }
              }
            }

            // Store the accumulated F values back to memory.
            // Boundary checks ensure only valid elements are written.
            for (int i_unroll = 0; i_unroll < UNROLL_I; ++i_unroll) {
              for (int j_unroll = 0; j_unroll < UNROLL_J; ++j_unroll) {
                if (c1 + i_unroll < c1_end_tile && c2 + j_unroll < c2_end_tile) {
                  F[c1 + i_unroll][c2 + j_unroll] = f_acc[i_unroll][j_unroll];
                }
              }
            }
          } // End of c2 (unrolled) loop
        } // End of c1 (unrolled) loop
      } // End of c5_outer (tile) loop
    } // End of c2_outer (tile) loop
  } // End of c1_outer (tile) loop
}
