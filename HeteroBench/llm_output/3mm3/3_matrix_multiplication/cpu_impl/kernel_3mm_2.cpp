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
#include <immintrin.h> // For AVX intrinsics
#include <algorithm>   // For std::min

// Define tile sizes for cache optimization.
// These values can be tuned based on target architecture's cache sizes.
// For example, 64x64 doubles (32KB) fits well within a typical 32KB L1D cache.
#ifndef TILE_NI
#define TILE_NI 64
#endif
#ifndef TILE_NL
#define TILE_NL 64
#endif

// Vectorization factor for doubles using AVX2 (4 doubles per __m256d register).
// For AVX-512, this would be 8.
#define VEC_LEN 4

void kernel_3m_2(double E[NI + 0][NJ + 0], double F[NJ + 0][NL + 0], double G[NI + 0][NL + 0])
{
  int c1, c2, c6;
  int c1_t, c6_t; // Loop variables for tiling

  // Loop tiling for G (result matrix) to improve cache reuse.
  // The outer loops iterate over blocks of G.
  for (c1_t = 0; c1_t < NI; c1_t += TILE_NI) {
    for (c6_t = 0; c6_t < NL; c6_t += TILE_NL) {
      // The c2 loop (summation index) is placed outside the inner tiled loops.
      // This ensures that for each block of G, contributions from all c2 values
      // are accumulated before the block is potentially evicted from cache.
      for (c2 = 0; c2 < NJ; c2++) {
        // Inner loops iterate within the current tile.
        // std::min is used to handle matrix dimensions that are not exact multiples of tile sizes.
        for (c1 = c1_t; c1 < std::min(c1_t + TILE_NI, NI); c1++) {
          // E[c1][c2] is a scalar value that is constant for the innermost c6 loop.
          // Broadcast this scalar to all elements of a vector register for efficient SIMD operations.
          __m256d e_val_vec = _mm256_set1_pd(E[c1][c2]);

          // Determine the end boundary for the inner c6 loop within the current tile.
          int c6_vec_end = std::min(c6_t + TILE_NL, NL);

          // Vectorized loop for c6. Processes VEC_LEN (4) doubles at a time using AVX2 intrinsics.
          // _mm256_loadu_pd and _mm256_storeu_pd are used for unaligned memory access,
          // as array pointers are not guaranteed to be 32-byte aligned.
          for (c6 = c6_t; c6 <= c6_vec_end - VEC_LEN; c6 += VEC_LEN) {
            __m256d f_vec = _mm256_loadu_pd(&F[c2][c6]); // Load VEC_LEN doubles from F
            __m256d g_vec = _mm256_loadu_pd(&G[c1][c6]); // Load VEC_LEN doubles from G

            // Perform the multiply-add operation: g_vec += e_val_vec * f_vec
            g_vec = _mm256_add_pd(g_vec, _mm256_mul_pd(e_val_vec, f_vec));

            _mm256_storeu_pd(&G[c1][c6], g_vec); // Store the result back to G
          }

          // Remainder loop for c6 (scalar operations).
          // This handles any elements that could not be processed by the vectorized loop
          // because the remaining count is less than VEC_LEN.
          for (; c6 < c6_vec_end; c6++) {
            G[c1][c6] += E[c1][c2] * F[c2][c6];
          }
        }
      }
    }
  }
}