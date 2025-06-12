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
// The original code included "omp.h" but did not use OpenMP directives.
// For single-threaded optimization, it's not needed.
// #include "omp.h" 
#include <cstring>
#include <iostream>
#include <math.h>
#include <immintrin.h> // Required for AVX/AVX2 intrinsics

// It's generally good practice to avoid 'using namespace std;' in headers or large scopes.
// using namespace std; 

void double_thresholding(double *suppressed_image, int height, int width,
                  int high_threshold, int low_threshold,
                  uint8_t *outImage) {

    // Convert integer thresholds to double for SIMD comparisons.
    // This avoids repeated implicit conversions inside the loop and ensures correct floating-point comparisons.
    const double high_thresh_d = static_cast<double>(high_threshold);
    const double low_thresh_d = static_cast<double>(low_threshold);

    // Broadcast threshold values into AVX registers.
    // These registers will hold the same threshold value across all 4 double lanes.
    const __m256d v_high_threshold = _mm256_set1_pd(high_thresh_d);
    const __m256d v_low_threshold = _mm256_set1_pd(low_thresh_d);

    // Broadcast output values (0, 100, 255) into AVX registers as doubles.
    // These will be used in blend operations to construct the result vector.
    const __m256d v_255_d = _mm256_set1_pd(255.0);
    const __m256d v_100_d = _mm256_set1_pd(100.0);
    const __m256d v_0_d = _mm256_set1_pd(0.0);

    // Loop reordering: Iterate rows in the outer loop and columns in the inner loop.
    // The original code used a column-major access pattern (col then row), which can lead to
    // cache misses if 'width' is large. This reordering to row-major access (row then col)
    // ensures contiguous memory access for `suppressed_image` and `outImage` within the
    // inner loop, significantly improving cache performance and enabling efficient vectorization.
    for (int row = 0; row < height; ++row) {
        // Calculate the starting index for the current row.
        // This avoids repeated multiplication inside the inner loop (strength reduction).
        int row_start_idx = row * width;

        // Vectorized loop: Process 8 double values at a time.
        // An AVX register (`__m256d`) holds 4 doubles. We use two AVX registers to process 8 doubles.
        // This allows for efficient packing of 8 resulting uint8_t values into a single 64-bit store.
        int col = 0;
        for (; col + 7 < width; col += 8) {
            // Calculate the base index for the current 8-element chunk.
            int pxIndex = row_start_idx + col;

            // Load 8 double values from `suppressed_image` using unaligned loads.
            // `_mm256_loadu_pd` is used as alignment of `suppressed_image` is not guaranteed.
            __m256d s_val_lo = _mm256_loadu_pd(&suppressed_image[pxIndex]);     // Loads suppressed_image[pxIndex] to [pxIndex+3]
            __m256d s_val_hi = _mm256_loadu_pd(&suppressed_image[pxIndex + 4]); // Loads suppressed_image[pxIndex+4] to [pxIndex+7]

            // Perform comparisons for the lower 4 doubles (s_val_lo).
            // `_CMP_GT_OQ` (Greater Than, Ordered, Quiet) is a standard comparison predicate for floating-point.
            __m256d mask_high_lo = _mm256_cmp_pd(s_val_lo, v_high_threshold, _CMP_GT_OQ);
            __m256d mask_low_lo = _mm256_cmp_pd(s_val_lo, v_low_threshold, _CMP_GT_OQ);

            // Perform comparisons for the upper 4 doubles (s_val_hi).
            __m256d mask_high_hi = _mm256_cmp_pd(s_val_hi, v_high_threshold, _CMP_GT_OQ);
            __m256d mask_low_hi = _mm256_cmp_pd(s_val_hi, v_low_threshold, _CMP_GT_OQ);

            // Apply the thresholding logic using blend operations for s_val_lo:
            // The logic is: if (val > high) 255 else if (val > low) 100 else 0.
            // This is implemented by starting with 0, then blending 100 if > low,
            // then blending 255 if > high (which correctly overwrites 100 for strong edges).
            __m256d result_lo = v_0_d;
            result_lo = _mm256_blendv_pd(result_lo, v_100_d, mask_low_lo);
            result_lo = _mm256_blendv_pd(result_lo, v_255_d, mask_high_lo);

            // Apply the same thresholding logic for s_val_hi:
            __m256d result_hi = v_0_d;
            result_hi = _mm256_blendv_pd(result_hi, v_100_d, mask_low_hi);
            result_hi = _mm256_blendv_pd(result_hi, v_255_d, mask_high_hi);

            // Convert the double results to 32-bit integers.
            // `_mm256_cvttpd_epi32` truncates the double values.
            // Each `__m256i` result contains 4 integers in its lower 128-bit lane, with the upper 128 bits zeroed.
            __m256i i_res_lo = _mm256_cvttpd_epi32(result_lo);
            __m256i i_res_hi = _mm256_cvttpd_epi32(result_hi);

            // Extract the lower 128-bit lanes from the `__m256i` results.
            // These `__m128i` registers now hold the 4 32-bit integers each.
            __m128i i_res_lo_128 = _mm256_extractf128_si256(i_res_lo, 0);
            __m128i i_res_hi_128 = _mm256_extractf128_si256(i_res_hi, 0);

            // Pack the two `__m128i` vectors (each with 4 32-bit ints) into a single `__m128i` vector
            // containing 8 16-bit shorts.
            __m128i packed_shorts = _mm_pack_epi32(i_res_lo_128, i_res_hi_128);

            // Pack the 8 16-bit shorts into 8-bit bytes using unsigned saturation.
            // `_mm_packus_epi16` packs 16 shorts into 16 bytes. By passing `packed_shorts` twice,
            // we effectively pack the lower 8 shorts into the lower 8 bytes of the result,
            // and the upper (zeroed) 8 shorts into the upper 8 bytes.
            __m128i packed_bytes = _mm_packus_epi16(packed_shorts, packed_shorts);

            // Store the lower 64 bits (8 bytes) of the `packed_bytes` vector to `outImage`.
            // `_mm_storel_epi64` is used for unaligned store of 8 bytes.
            _mm_storel_epi64(reinterpret_cast<__m128i*>(&outImage[pxIndex]), packed_bytes);
        }

        // Tail loop: Process any remaining elements that couldn't be handled by the vectorized loop.
        // This ensures correctness for `width` values not perfectly divisible by 8.
        for (; col < width; ++col) {
            int pxIndex = row_start_idx + col;
            double val = suppressed_image[pxIndex];
            if (val > high_threshold)
                outImage[pxIndex] = 255;   // Strong edge
            else if (val > low_threshold)
                outImage[pxIndex] = 100;   // Weak edge
            else
                outImage[pxIndex] = 0;     // Not an edge
        }
    }
}