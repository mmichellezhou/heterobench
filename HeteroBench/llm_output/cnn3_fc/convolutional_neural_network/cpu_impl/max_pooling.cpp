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
#include <algorithm>   // For std::max
#include <immintrin.h> // For AVX intrinsics (e.g., __m256d, _mm256_loadu_pd, _mm256_max_pd, etc.)

// The original code included <iostream> and <math.h> and used `using namespace std;`.
// <iostream> is not needed for the function's logic.
// <math.h> is replaced by <algorithm> for `std::max`.
// `using namespace std;` is omitted for cleaner code and explicit `std::` qualification.

void max_pooling(double *max_pooling_input, int pool_size, int pool_stride, int input_h, int input_w, double *max_pooling_output) {
  // Calculate output dimensions based on input, pool_size, and pool_stride.
  int output_h = (input_h - pool_size) / pool_stride + 1;
  int output_w = (input_w - pool_size) / pool_stride + 1;

  // Define the vector size for AVX double precision (256-bit register holds 4 doubles).
  const int VEC_SIZE = 4; 

  // Iterate over each output pixel (i, j)
  for (int i = 0; i < output_h; i++) {
    // Pre-calculate the starting row index in the input array for the current pooling window.
    // This reduces redundant multiplications inside the inner loops.
    int input_row_start_i = i * pool_stride; 
    
    for (int j = 0; j < output_w; j++) {
      // Pre-calculate the starting column index in the input array for the current pooling window.
      // This also reduces redundant multiplications.
      int input_col_start_j = j * pool_stride;

      // Initialize max_val for the current output pixel.
      // IMPORTANT: The original C++ code initializes `max_val` to `0`.
      // This behavior is maintained for functional equivalence with the provided C++ implementation.
      // If input values can be negative and the true maximum (e.g., -2 in [-5, -2, -8]) is expected,
      // `std::numeric_limits<double>::lowest()` should be used instead for a more general solution.
      double max_val = 0.0;

      // Iterate over rows (k) within the pooling window
      for (int k = 0; k < pool_size; k++) {
        // Calculate the starting memory address for the current row of the pooling window.
        // This pointer arithmetic ensures cache-friendly contiguous access within the row.
        double* current_row_ptr = max_pooling_input + (input_row_start_i + k) * input_w + input_col_start_j;

        // Initialize a vector register for finding the maximum within the current row (k).
        // Set all elements to 0.0, consistent with the overall `max_val` initialization.
        __m256d row_current_max_vec = _mm256_setzero_pd(); // All elements set to 0.0

        int l = 0;
        // Process columns (l) within the pooling window using AVX SIMD instructions.
        // This loop processes `VEC_SIZE` (4) doubles at a time.
        for (; l + VEC_SIZE <= pool_size; l += VEC_SIZE) {
          // Load 4 doubles from memory into a vector register.
          // `_mm256_loadu_pd` is used for unaligned loads, which is safer as input array alignment
          // is not guaranteed. If data is known to be 32-byte aligned, `_mm256_load_pd` could be used.
          __m256d data_vec = _mm256_loadu_pd(current_row_ptr + l);
          
          // Compute the element-wise maximum between the current row's max vector and the loaded data.
          row_current_max_vec = _mm256_max_pd(row_current_max_vec, data_vec);
        }

        // Perform a horizontal reduction on `row_current_max_vec` to find the single maximum value
        // from the 4 doubles in the vector register.
        // Let hmax = [d0, d1, d2, d3]
        __m256d hmax = row_current_max_vec;
        
        // Step 1: Max across 128-bit lanes.
        // `_mm256_permute2f128_pd(hmax, hmax, 0x01)` swaps the two 128-bit lanes (high and low).
        // Example: [d0, d1, d2, d3] becomes [d2, d3, d0, d1].
        hmax = _mm256_max_pd(hmax, _mm256_permute2f128_pd(hmax, hmax, 0x01));
        // After this, hmax = [max(d0,d2), max(d1,d3), max(d0,d2), max(d1,d3)]

        // Step 2: Max across 64-bit elements within each 128-bit lane.
        // `_mm256_shuffle_pd(hmax, hmax, 0x05)` swaps the two doubles within each 128-bit lane.
        // Example: [x, y, z, w] becomes [y, x, w, z].
        hmax = _mm256_max_pd(hmax, _mm256_shuffle_pd(hmax, hmax, 0x05));
        // After this, hmax = [max(d0,d1,d2,d3), max(d1,d0,d3,d2), max(d0,d1,d2,d3), max(d1,d0,d3,d2)]
        // The maximum value of the original vector is now replicated in all elements of `hmax`.

        // Extract the maximum value from the first element of the vector.
        double current_row_max = _mm256_cvtsd_f64(hmax);

        // Process any remaining elements in the row (if `pool_size` is not a multiple of `VEC_SIZE`)
        // using scalar operations.
        for (; l < pool_size; l++) {
          current_row_max = std::max(current_row_max, current_row_ptr[l]);
        }
        
        // Update the overall `max_val` for the current output pixel by comparing it with the
        // maximum value found in the current row (`current_row_max`).
        max_val = std::max(max_val, current_row_max);
      }
      // Store the final maximum value for the current output pixel in the output array.
      max_pooling_output[i * output_w + j] = max_val;
    }
  }
}