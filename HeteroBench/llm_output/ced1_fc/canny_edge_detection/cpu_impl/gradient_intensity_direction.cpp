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
 
#include "cpu_impl.h" // Assumed to define OFFSET (typically 1 for 3x3 kernels)
#include "omp.h"      // Included in original, kept for compatibility, not used for single-threaded
#include <cstring>    // For memset
#include <iostream>   // Included in original, not used
#include <math.h>     // For sqrt, atan2, M_PI
#include <algorithm>  // For std::fill

// Using namespace std as in the original code
using namespace std;

void gradient_intensity_direction(const uint8_t *inImage, int height,
                                  int width, double *intensity,
                                  uint8_t *direction) {
  // Optimization: Fix for memset on double array.
  // The original `memset(intensity, 0.0, height*width);` was incorrect
  // as memset operates on bytes, not elements for non-byte types.
  // This correctly initializes the entire intensity array to 0.0.
  std::fill(intensity, intensity + height * width, 0.0);
  
  // This memset for uint8_t is correct and efficient.
  memset(direction, 0, height * width);

  // Optimization: Pre-calculate the constant for converting radians to degrees.
  // M_PI is a common extension in math.h, used in the original code.
  const double RAD_TO_DEG_FACTOR = 360.0 / (2.0 * M_PI);

  // Optimization: Swapped loop order for improved cache locality.
  // Iterating rows in the outer loop and columns in the inner loop
  // ensures more contiguous memory access for `inImage` (row-major order).
  for (int row = OFFSET; row < height - OFFSET; ++row) {
    // Optimization: Pre-calculate row base indices.
    // This avoids repeated multiplications (`row * width`) inside the inner loop.
    const int row_prev_idx = (row - 1) * width;
    const int row_curr_idx = row * width;
    const int row_next_idx = (row + 1) * width;

    for (int col = OFFSET; col < width - OFFSET; ++col) {
      // Calculate the linear index for the current pixel (center of the 3x3 window).
      const int pxIndex = col + row_curr_idx;

      // Optimization: Unroll the 3x3 convolution kernel.
      // The original Gx and Gy kernels are sparse (contain zeros),
      // allowing for strength reduction by eliminating multiplications by zero
      // and additions of zero. This also removes the inner `krow`, `kcol` loops
      // and `kIndex` variable, reducing loop overhead and array lookups.
      //
      // Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
      // Gy[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

      // Fetch the 9 surrounding pixel values.
      // Cast to double immediately to ensure floating-point arithmetic
      // throughout the sum calculation, preventing potential intermediate
      // integer overflows and maintaining precision.
      double val_nw = static_cast<double>(inImage[col - 1 + row_prev_idx]); // Top-left
      double val_n  = static_cast<double>(inImage[col + row_prev_idx]);     // Top-middle
      double val_ne = static_cast<double>(inImage[col + 1 + row_prev_idx]); // Top-right
      double val_w  = static_cast<double>(inImage[col - 1 + row_curr_idx]); // Middle-left
      // The center pixel (Gx[4], Gy[4]) has a kernel coefficient of 0, so it's not needed.
      double val_e  = static_cast<double>(inImage[col + 1 + row_curr_idx]); // Middle-right
      double val_sw = static_cast<double>(inImage[col - 1 + row_next_idx]); // Bottom-left
      double val_s  = static_cast<double>(inImage[col + row_next_idx]);     // Bottom-middle
      double val_se = static_cast<double>(inImage[col + 1 + row_next_idx]); // Bottom-right

      // Calculate Gx_sum using the unrolled and optimized expressions.
      double Gx_sum = -val_nw + val_ne - 2.0 * val_w + 2.0 * val_e - val_sw + val_se;
      
      // Calculate Gy_sum using the unrolled and optimized expressions.
      double Gy_sum = val_nw + 2.0 * val_n + val_ne - val_sw - 2.0 * val_s - val_se;

      // Optimization: Corrected the condition for zero gradient.
      // The original `||` (OR) would set intensity/direction to zero if *either* Gx_sum or Gy_sum was zero,
      // which is mathematically incorrect for gradient calculation (e.g., a purely horizontal or vertical
      // gradient would be incorrectly zeroed).
      // A zero gradient (and thus zero intensity and direction) only occurs when *both* components are zero.
      if (Gx_sum == 0.0 && Gy_sum == 0.0) {
        intensity[pxIndex] = 0.0;
        direction[pxIndex] = 0;
      } else {
        // Calculate intensity (magnitude of the gradient vector).
        intensity[pxIndex] = std::sqrt((Gx_sum * Gx_sum) + (Gy_sum * Gy_sum));
        
        // Calculate direction (angle of the gradient vector).
        double theta = std::atan2(Gy_sum, Gx_sum);
        theta *= RAD_TO_DEG_FACTOR; // Convert radians to degrees.

        // Determine direction quadrant based on the calculated angle.
        // These conditions are directly from the original code and define the 4 directions.
        if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) || (theta >= 157.5))
          direction[pxIndex] = 1;  // horizontal -
        else if ((theta > 22.5 && theta <= 67.5) ||
                 (theta > -157.5 && theta <= -112.5))
          direction[pxIndex] = 2;  // north-east -> south-west /
        else if ((theta > 67.5 && theta <= 112.5) ||
                 (theta >= -112.5 && theta < -67.5))
          direction[pxIndex] = 3;  // vertical |
        else if ((theta >= -67.5 && theta < -22.5) ||
                 (theta > 112.5 && theta < 157.5))
          direction[pxIndex] = 4;  // north-west -> south-east \'
      }
    }
  }
}