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

// Define a tile size for loop blocking.
// A common choice for 'double' precision on modern CPUs is 32 or 64.
// A 64x64 block of doubles occupies 64*64*8 = 32768 bytes (32KB),
// which typically fits within L1 data cache. This helps improve data locality.
#ifndef TILE_SIZE
#define TILE_SIZE 64 
#endif

using namespace std;

void kernel_3m_0(double A[NI + 0][NK + 0], double B[NK + 0][NJ + 0], double E[NI + 0][NJ + 0])
{
  int c1, c2, c5;
  int c1_block, c2_block, c5_block;

  // Optimized loop order: c1 (NI), c5 (NK), c2 (NJ) (ikj order)
  // The original loop order (ijk) caused column-wise access to matrix B (B[c5][c2]),
  // leading to poor cache performance due to strided memory access.
  // This 'ikj' order ensures that B[c5][c2] is accessed row-wise in the innermost loop,
  // improving spatial locality and cache utilization. E[c1][c2] is also accessed row-wise.

  // Loop tiling (blocking) to further improve cache reuse for larger matrices.
  // The outer loops iterate over blocks of the matrices.
  for (c1_block = 0; c1_block < NI; c1_block += TILE_SIZE) {
    for (c5_block = 0; c5_block < NK; c5_block += TILE_SIZE) {
      for (c2_block = 0; c2_block < NJ; c2_block += TILE_SIZE) {

        // Inner loops process elements within the current tile.
        // The loop bounds handle cases where dimensions are not exact multiples of TILE_SIZE.
        for (c1 = c1_block; c1 < (c1_block + TILE_SIZE > NI ? NI : c1_block + TILE_SIZE); c1++) {
          for (c5 = c5_block; c5 < (c5_block + TILE_SIZE > NK ? NK : c5_block + TILE_SIZE); c5++) {
            // Load A[c1][c5] once per inner c2 loop.
            // This value is constant for the innermost loop, allowing for efficient
            // scalar-vector multiplication by the compiler's auto-vectorizer.
            double val_A = A[c1][c5];

            // Innermost loop: highly amenable to auto-vectorization (SIMD).
            // E[c1][c2] and B[c5][c2] are accessed contiguously, which is ideal for SIMD.
            for (c2 = c2_block; c2 < (c2_block + TILE_SIZE > NJ ? NJ : c2_block + TILE_SIZE); c2++) {
              E[c1][c2] += val_A * B[c5][c2];
            }
          }
        }
      }
    }
  }
}