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
#include <cstring>     // For memcpy
#include <immintrin.h> // For AVX2 intrinsics

// Assuming OFFSET is 1 for a 3x3 kernel, as implied by neighbor access patterns.
// If OFFSET is defined elsewhere (e.g., in cpu_impl.h or via compiler flags),
// this definition will be ignored.
#ifndef OFFSET
#define OFFSET 1
#endif

void hysteresis(uint8_t *inImage, int height, int width,
                uint8_t *outImage) {
  // 1. Initial copy of the image data.
  // This is already an optimized operation provided by the standard library.
  memcpy(outImage, inImage, (size_t)width * height * sizeof(uint8_t));

  // Constants for SIMD operations.
  // These are loaded once and reused, avoiding repeated memory access.
  const __m256i v_100 = _mm256_set1_epi8(100); // Vector of 32 bytes, all 100
  const __m256i v_255 = _mm256_set1_epi8(255); // Vector of 32 bytes, all 255
  const __m256i v_0 = _mm256_setzero_si256();  // Vector of 32 bytes, all 0

  // Loop over rows (outer loop) for better cache locality.
  // This ensures that memory accesses for p_prev_row, p_curr_row, p_next_row
  // are mostly sequential within the inner loop.
  // The loop bounds (OFFSET to height - OFFSET) ensure that
  // row-1 and row+1 accesses are within image bounds.
  for (int row = OFFSET; row < height - OFFSET; ++row) {
    // Pointers to the current, previous, and next rows for efficient access.
    // These pointers are calculated once per row.
    uint8_t *p_prev_row = outImage + (row - 1) * width;
    uint8_t *p_curr_row = outImage + row * width;
    uint8_t *p_next_row = outImage + (row + 1) * width;

    // Process columns with AVX2 SIMD instructions (32 bytes/pixels at a time).
    // The loop limit `width - OFFSET - 32` ensures that all required neighbor loads
    // (e.g., `p_curr_row + col + 1` which accesses up to `col + 1 + 31 = col + 32`)
    // stay within the valid processing region `[OFFSET, width - OFFSET - 1]`.
    // Specifically, the rightmost pixel accessed by a neighbor load is `col + 32`.
    // This must be less than `width - OFFSET`. So `col + 32 < width - OFFSET`,
    // which means `col < width - OFFSET - 32`.
    // The loop condition `col <= width - OFFSET - 32` correctly handles this.
    int col = OFFSET;
    for (; col <= width - OFFSET - 32; col += 32) {
      // Load the current 32 pixels from the output image.
      __m256i v_curr_pixels = _mm256_loadu_si256((__m256i*)(p_curr_row + col));

      // Create a mask for pixels that are equal to 100.
      // `_mm256_cmpeq_epi8` sets bytes to 0xFF if equal, 0x00 otherwise.
      __m256i v_mask_is_100 = _mm256_cmpeq_epi8(v_curr_pixels, v_100);

      // Load all 8 sets of neighbor pixels for the current 32-pixel block.
      // `_mm256_loadu_si256` performs unaligned loads, which is suitable here
      // as the addresses `col-1`, `col+1`, etc., might not be 32-byte aligned.
      __m256i v_n_left        = _mm256_loadu_si256((__m256i*)(p_curr_row + col - 1));
      __m256i v_n_right       = _mm256_loadu_si256((__m256i*)(p_curr_row + col + 1));
      __m256i v_n_top         = _mm256_loadu_si256((__m256i*)(p_prev_row + col));
      __m256i v_n_bottom      = _mm256_loadu_si256((__m256i*)(p_next_row + col));
      __m256i v_n_top_left    = _mm256_loadu_si256((__m256i*)(p_prev_row + col - 1));
      __m256i v_n_top_right   = _mm256_loadu_si256((__m256i*)(p_prev_row + col + 1));
      __m256i v_n_bottom_left = _mm256_loadu_si256((__m256i*)(p_next_row + col - 1));
      __m256i v_n_bottom_right= _mm256_loadu_si256((__m256i*)(p_next_row + col + 1));

      // Compare each neighbor set with 255 to create masks.
      // Each resulting vector will have 0xFF for bytes where the neighbor was 255, 0x00 otherwise.
      __m256i v_res_left        = _mm256_cmpeq_epi8(v_n_left, v_255);
      __m256i v_res_right       = _mm256_cmpeq_epi8(v_n_right, v_255);
      __m256i v_res_top         = _mm256_cmpeq_epi8(v_n_top, v_255);
      __m256i v_res_bottom      = _mm256_cmpeq_epi8(v_n_bottom, v_255);
      __m256i v_res_top_left    = _mm256_cmpeq_epi8(v_n_top_left, v_255);
      __m256i v_res_top_right   = _mm256_cmpeq_epi8(v_n_top_right, v_255);
      __m256i v_res_bottom_left = _mm256_cmpeq_epi8(v_n_bottom_left, v_255);
      __m256i v_res_bottom_right= _mm256_cmpeq_epi8(v_n_bottom_right, v_255);

      // Combine all neighbor masks using bitwise OR to find if ANY neighbor is 255.
      // If any of the 8 neighbor masks has a 0xFF at a given byte position,
      // `v_any_neighbor_255` will have 0xFF at that position.
      __m256i v_any_neighbor_255 = _mm256_or_si256(v_res_left, v_res_right);
      v_any_neighbor_255 = _mm256_or_si256(v_any_neighbor_255, v_res_top);
      v_any_neighbor_255 = _mm256_or_si256(v_any_neighbor_255, v_res_bottom);
      v_any_neighbor_255 = _mm256_or_si256(v_any_neighbor_255, v_res_top_left);
      v_any_neighbor_255 = _mm256_or_si256(v_any_neighbor_255, v_res_top_right);
      v_any_neighbor_255 = _mm256_or_si256(v_any_neighbor_255, v_res_bottom_left);
      v_any_neighbor_255 = _mm256_or_si256(v_any_neighbor_255, v_res_bottom_right);

      // Calculate mask for pixels that should become 255:
      // This happens if the current pixel was 100 AND any neighbor was 255.
      __m256i v_mask_set_255 = _mm256_and_si256(v_mask_is_100, v_any_neighbor_255);

      // Calculate mask for pixels that should become 0:
      // This happens if the current pixel was 100 AND NO neighbor was 255.
      // `_mm256_andnot_si256(A, B)` computes `(~A) & B`.
      // So, `(~v_any_neighbor_255) & v_mask_is_100`.
      __m256i v_mask_set_0 = _mm256_andnot_si256(v_any_neighbor_255, v_mask_is_100);

      // Initialize final pixels with their current values.
      __m256i v_final_pixels = v_curr_pixels;

      // Blend in 255 where `v_mask_set_255` is true (0xFF).
      // If the mask byte is 0xFF, the corresponding byte from `v_255` is chosen.
      // Otherwise, the corresponding byte from `v_final_pixels` (its current value) is kept.
      v_final_pixels = _mm256_blendv_epi8(v_final_pixels, v_255, v_mask_set_255);

      // Blend in 0 where `v_mask_set_0` is true (0xFF).
      // This operation is applied after the previous blend.
      // If the mask byte is 0xFF, the corresponding byte from `v_0` is chosen.
      // Otherwise, the corresponding byte from `v_final_pixels` (its value after the first blend) is kept.
      v_final_pixels = _mm256_blendv_epi8(v_final_pixels, v_0, v_mask_set_0);

      // Store the computed pixels back to `outImage`.
      _mm256_storeu_si256((__m256i*)(p_curr_row + col), v_final_pixels);
    }

    // Scalar remainder loop for columns not covered by SIMD.
    // This handles the last few pixels in a row if `width - OFFSET` is not a multiple of 32.
    for (; col < width - OFFSET; ++col) {
      int pxIndex = col + (row * width);
      if (outImage[pxIndex] == 100) {
        if (outImage[pxIndex - 1] == 255 ||
            outImage[pxIndex + 1] == 255 ||
            outImage[pxIndex - width] == 255 ||
            outImage[pxIndex + width] == 255 ||
            outImage[pxIndex - width - 1] == 255 ||
            outImage[pxIndex - width + 1] == 255 ||
            outImage[pxIndex + width - 1] == 255 ||
            outImage[pxIndex + width + 1] == 255)
          outImage[pxIndex] = 255;
        else
          outImage[pxIndex] = 0;
      }
    }
  }
}