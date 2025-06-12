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
 
#include "cpu_impl.h" // Assumed to define OFFSET, typically 1 for a 3x3 kernel
#include "omp.h"      // Kept as in original, though not used for single-threaded optimization
#include <cstring>
#include <iostream>
#include <math.h>     // Kept as in original, though not strictly needed for this kernel

using namespace std;

void gaussian_filter(const uint8_t *inImage, int height, int width,
                   uint8_t *outImage) {
  // The Gaussian kernel values are fixed for a 3x3 filter.
  // Keeping it as a local const array as in the original.
  const double kernel[9] = {0.0625, 0.125, 0.0625, 0.1250, 0.250, 0.1250, 0.0625, 0.125, 0.0625};

  // Initialize the entire output image to 0.
  // This correctly handles the border pixels which are not processed by the main loops,
  // maintaining functional equivalence with the original code's behavior.
  memset(outImage, 0, (size_t)height * width);

  // Optimized loop order: Iterate rows in the outer loop, columns in the inner loop.
  // This improves cache locality significantly because image data is typically stored
  // in row-major order, meaning pixels within a row are contiguous in memory.
  // Accessing `inImage` and writing to `outImage` contiguously in the inner loop
  // leads to better cache utilization and fewer cache misses.
  for (int row = OFFSET; row < height - OFFSET; row++) {
    // Pre-calculate row offsets for the 3x3 kernel relative to the current row.
    // This avoids repeated multiplications by `width` inside the inner column loop (strength reduction).
    const int row_offset_m1 = (row - 1) * width; // Offset for the row above (krow = -1)
    const int row_offset_0  = row * width;       // Offset for the current row (krow = 0)
    const int row_offset_p1 = (row + 1) * width; // Offset for the row below (krow = 1)

    for (int col = OFFSET; col < width - OFFSET; col++) {
      double outIntensity = 0;

      // Calculate the base index for the current output pixel.
      // This corresponds to the center pixel of the 3x3 convolution window.
      const int pxIndex = col + row_offset_0;

      // Unroll the 3x3 kernel convolution loop completely.
      // This eliminates loop overhead (loop control variables, conditions, increments)
      // and exposes more instruction-level parallelism. It also makes the code
      // more amenable to compiler auto-vectorization (SIMD instructions)
      // by presenting a fixed sequence of operations.

      // Apply kernel for krow = -1 (top row of the 3x3 kernel)
      outIntensity += (double)inImage[col - 1 + row_offset_m1] * kernel[0]; // kcol=-1
      outIntensity += (double)inImage[col + 0 + row_offset_m1] * kernel[1]; // kcol=0
      outIntensity += (double)inImage[col + 1 + row_offset_m1] * kernel[2]; // kcol=1

      // Apply kernel for krow = 0 (middle row of the 3x3 kernel)
      outIntensity += (double)inImage[col - 1 + row_offset_0] * kernel[3]; // kcol=-1
      outIntensity += (double)inImage[col + 0 + row_offset_0] * kernel[4]; // kcol=0 (center pixel)
      outIntensity += (double)inImage[col + 1 + row_offset_0] * kernel[5]; // kcol=1

      // Apply kernel for krow = 1 (bottom row of the 3x3 kernel)
      outIntensity += (double)inImage[col - 1 + row_offset_p1] * kernel[6]; // kcol=-1
      outIntensity += (double)inImage[col + 0 + row_offset_p1] * kernel[7]; // kcol=0
      outIntensity += (double)inImage[col + 1 + row_offset_p1] * kernel[8]; // kcol=1

      // Store the calculated intensity, cast back to uint8_t.
      // This maintains the original data type conversion behavior.
      outImage[pxIndex] = (uint8_t)(outIntensity);
    }
  }
}