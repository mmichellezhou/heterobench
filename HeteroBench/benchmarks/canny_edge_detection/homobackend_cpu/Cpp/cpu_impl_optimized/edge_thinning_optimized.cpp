#include "cpu_impl.h"

void edge_thinning_optimized(double *intensity, uint8_t *direction, int height,
                             int width, double *outImage) {
  // 1. Initial memcpy: Copies 'intensity' to 'outImage'.
  // This is necessary because some pixels are unconditionally suppressed,
  // and others might remain unchanged if the 'if' condition in the switch is false.
  // Using (size_t) for multiplication to prevent potential overflow for large images.
  memcpy(outImage, intensity, (size_t)width * height * sizeof(double));

  // Pre-calculate loop invariants and common offsets for strength reduction
  const int width_minus_1 = width - 1;
  const int width_plus_1 = width + 1;

  // Define the core processing region where the switch statement applies.
  // This region excludes the 1-pixel wide border that is always zeroed out
  // according to the original logic's `if` condition.
  const int core_start_row = OFFSET + 1;
  const int core_end_row = height - OFFSET - 1; // Exclusive upper bound for loop
  const int core_start_col = OFFSET + 1;
  const int core_end_col = width - OFFSET - 1; // Exclusive upper bound for loop

  // Unrolling factor for the inner loop to expose Instruction-Level Parallelism (ILP).
  // Since vectorization is disallowed, unrolling helps the CPU's out-of-order
  // execution engine find independent operations.
  const int UNROLL_FACTOR = 4;

  // Process the core region (where the switch logic applies).
  // Loop order: row (outer), col (inner) for better cache locality.
  // This ensures contiguous memory accesses for `intensity` and `direction` arrays
  // in the inner loop.
  for (int row = core_start_row; row < core_end_row; ++row) {
    // Pre-calculate row_offset for strength reduction in pxIndex calculation.
    int row_offset = row * width;

    // Main unrolled loop for columns.
    // The loop condition ensures we don't go out of bounds when unrolling.
    for (int col = core_start_col; col < core_end_col - (UNROLL_FACTOR - 1); col += UNROLL_FACTOR) {
      // Process pixel at 'col'
      int pxIndex0 = row_offset + col;
      double current_intensity0 = intensity[pxIndex0];
      uint8_t current_direction0 = direction[pxIndex0];
      switch (current_direction0) {
      case 1:
        if (intensity[pxIndex0 - 1] >= current_intensity0 || intensity[pxIndex0 + 1] > current_intensity0)
          outImage[pxIndex0] = 0.0;
        break;
      case 2:
        if (intensity[pxIndex0 - width_minus_1] >= current_intensity0 || intensity[pxIndex0 + width_minus_1] > current_intensity0)
          outImage[pxIndex0] = 0.0;
        break;
      case 3:
        if (intensity[pxIndex0 - width] >= current_intensity0 || intensity[pxIndex0 + width] > current_intensity0)
          outImage[pxIndex0] = 0.0;
        break;
      case 4:
        if (intensity[pxIndex0 - width_plus_1] >= current_intensity0 || intensity[pxIndex0 + width_plus_1] > current_intensity0)
          outImage[pxIndex0] = 0.0;
        break;
      default:
        outImage[pxIndex0] = 0.0;
        break;
      }

      // Process pixel at 'col + 1'
      int pxIndex1 = row_offset + col + 1;
      double current_intensity1 = intensity[pxIndex1];
      uint8_t current_direction1 = direction[pxIndex1];
      switch (current_direction1) {
      case 1:
        if (intensity[pxIndex1 - 1] >= current_intensity1 || intensity[pxIndex1 + 1] > current_intensity1)
          outImage[pxIndex1] = 0.0;
        break;
      case 2:
        if (intensity[pxIndex1 - width_minus_1] >= current_intensity1 || intensity[pxIndex1 + width_minus_1] > current_intensity1)
          outImage[pxIndex1] = 0.0;
        break;
      case 3:
        if (intensity[pxIndex1 - width] >= current_intensity1 || intensity[pxIndex1 + width] > current_intensity1)
          outImage[pxIndex1] = 0.0;
        break;
      case 4:
        if (intensity[pxIndex1 - width_plus_1] >= current_intensity1 || intensity[pxIndex1 + width_plus_1] > current_intensity1)
          outImage[pxIndex1] = 0.0;
        break;
      default:
        outImage[pxIndex1] = 0.0;
        break;
      }

      // Process pixel at 'col + 2'
      int pxIndex2 = row_offset + col + 2;
      double current_intensity2 = intensity[pxIndex2];
      uint8_t current_direction2 = direction[pxIndex2];
      switch (current_direction2) {
      case 1:
        if (intensity[pxIndex2 - 1] >= current_intensity2 || intensity[pxIndex2 + 1] > current_intensity2)
          outImage[pxIndex2] = 0.0;
        break;
      case 2:
        if (intensity[pxIndex2 - width_minus_1] >= current_intensity2 || intensity[pxIndex2 + width_minus_1] > current_intensity2)
          outImage[pxIndex2] = 0.0;
        break;
      case 3:
        if (intensity[pxIndex2 - width] >= current_intensity2 || intensity[pxIndex2 + width] > current_intensity2)
          outImage[pxIndex2] = 0.0;
        break;
      case 4:
        if (intensity[pxIndex2 - width_plus_1] >= current_intensity2 || intensity[pxIndex2 + width_plus_1] > current_intensity2)
          outImage[pxIndex2] = 0.0;
        break;
      default:
        outImage[pxIndex2] = 0.0;
        break;
      }

      // Process pixel at 'col + 3'
      int pxIndex3 = row_offset + col + 3;
      double current_intensity3 = intensity[pxIndex3];
      uint8_t current_direction3 = direction[pxIndex3];
      switch (current_direction3) {
      case 1:
        if (intensity[pxIndex3 - 1] >= current_intensity3 || intensity[pxIndex3 + 1] > current_intensity3)
          outImage[pxIndex3] = 0.0;
        break;
      case 2:
        if (intensity[pxIndex3 - width_minus_1] >= current_intensity3 || intensity[pxIndex3 + width_minus_1] > current_intensity3)
          outImage[pxIndex3] = 0.0;
        break;
      case 3:
        if (intensity[pxIndex3 - width] >= current_intensity3 || intensity[pxIndex3 + width] > current_intensity3)
          outImage[pxIndex3] = 0.0;
        break;
      case 4:
        if (intensity[pxIndex3 - width_plus_1] >= current_intensity3 || intensity[pxIndex3 + width_plus_1] > current_intensity3)
          outImage[pxIndex3] = 0.0;
        break;
      default:
        outImage[pxIndex3] = 0.0;
        break;
      }
    }

    // Remainder loop for columns not covered by unrolling.
    // This ensures all pixels in the core region are processed.
    for (int col = core_end_col - ((core_end_col - core_start_col) % UNROLL_FACTOR); col < core_end_col; ++col) {
      int pxIndex = row_offset + col;
      double current_intensity = intensity[pxIndex];
      uint8_t current_direction = direction[pxIndex];
      switch (current_direction) {
      case 1:
        if (intensity[pxIndex - 1] >= current_intensity || intensity[pxIndex + 1] > current_intensity)
          outImage[pxIndex] = 0.0;
        break;
      case 2:
        if (intensity[pxIndex - width_minus_1] >= current_intensity || intensity[pxIndex + width_minus_1] > current_intensity)
          outImage[pxIndex] = 0.0;
        break;
      case 3:
        if (intensity[pxIndex - width] >= current_intensity || intensity[pxIndex + width] > current_intensity)
          outImage[pxIndex] = 0.0;
        break;
      case 4:
        if (intensity[pxIndex - width_plus_1] >= current_intensity || intensity[pxIndex + width_plus_1] > current_intensity)
          outImage[pxIndex] = 0.0;
        break;
      default:
        outImage[pxIndex] = 0.0;
        break;
      }
    }
  }

  // Explicitly zero out the border pixels that were copied by memcpy
  // but should be 0 according to the original logic's unconditional suppression.
  // This avoids the `if` branch inside the main processing loop.

  const int border_col_start = OFFSET;
  const int border_col_end = width - OFFSET; // Exclusive upper bound

  // Zero out the top border row within the processing region (row = OFFSET)
  int top_row_offset = OFFSET * width;
  for (int col = border_col_start; col < border_col_end; ++col) {
    outImage[top_row_offset + col] = 0.0;
  }

  // Zero out the bottom border row within the processing region (row = height - OFFSET - 1)
  int bottom_row_offset = (height - OFFSET - 1) * width;
  for (int col = border_col_start; col < border_col_end; ++col) {
    outImage[bottom_row_offset + col] = 0.0;
  }

  // Zero out the left and right border columns within the processing region,
  // excluding the corner pixels already handled by the top/bottom row zeroing.
  // Loop rows from OFFSET + 1 to height - OFFSET - 2 (which is core_start_row to core_end_row - 1)
  for (int row = core_start_row; row < core_end_row; ++row) {
    int row_offset = row * width;
    outImage[row_offset + OFFSET] = 0.0; // Left border column
    outImage[row_offset + (width - OFFSET - 1)] = 0.0; // Right border column
  }
}