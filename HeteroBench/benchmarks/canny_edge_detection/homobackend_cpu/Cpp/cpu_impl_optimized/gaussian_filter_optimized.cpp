#include "cpu_impl.h"

void gaussian_filter_optimized(const uint8_t *inImage, int height, int width,
                               uint8_t *outImage) {
  // The original kernel array is available and its values are used directly
  // in the unrolled loop for optimal register usage and reduced memory access.
  const double kernel[9] = {0.0625, 0.125, 0.0625, 0.1250, 0.250, 0.1250, 0.0625, 0.125, 0.0625};

  // The original code initializes the entire output image to zero using memset.
  // This means the border regions (defined by OFFSET) are not computed by the
  // convolution loops and remain zero. This behavior is preserved for
  // functional equivalence.
  memset(outImage, 0, height * width);

  // Pre-calculate width-related offsets for the kernel accesses.
  // This is a form of strength reduction, avoiding repeated calculations
  // of `krow * width` within the innermost loop.
  // Assuming OFFSET is 1 for a 3x3 kernel, these correspond to:
  // - (width + 1) for top-left
  // - width for top-middle
  // - (width - 1) for top-right
  // - 1 for middle-right
  // + (width - 1) for bottom-left
  // + width for bottom-middle
  // + (width + 1) for bottom-right
  const int width_minus_1 = width - 1;
  const int width_plus_1 = width + 1;
  const int width_val = width; // Represents 0 * width for the middle row of the kernel

  // Loop Reordering:
  // The original loops iterate `col` then `row`. Swapping them to `row` then `col`
  // improves cache locality. In row-major image data, iterating `col` in the
  // innermost loop means `pxIndex` increments by 1, leading to contiguous
  // memory accesses for `outImage` writes and `inImage` reads, which is
  // highly cache-friendly.
  for (int row = OFFSET; row < height - OFFSET; ++row) {
    // Strength Reduction:
    // Calculate the base offset for the current row once per row iteration.
    // `row * width` is constant for all columns in a given row.
    const int row_offset_base = row * width;

    for (int col = OFFSET; col < width - OFFSET; ++col) {
      double outIntensity = 0.0;
      // Calculate the current pixel index in the image.
      const int pxIndex = col + row_offset_base;

      // Loop Unrolling and Register Optimization:
      // The 3x3 kernel application loops (`krow`, `kcol`) are fully unrolled.
      // This eliminates loop overhead (branching, index increments like `kIndex++`)
      // and exposes all 9 multiply-add operations directly to the compiler.
      // This allows the CPU to exploit Instruction-Level Parallelism (ILP)
      // by scheduling these independent operations concurrently, leading to
      // better utilization of execution units.
      // The `kernel` values are constants and will likely be loaded into
      // floating-point registers once by the compiler. `outIntensity` will
      // also reside in a register.

      // krow = -OFFSET (equivalent to krow = -1 for a 3x3 kernel)
      outIntensity += static_cast<double>(inImage[pxIndex - width_plus_1]) * kernel[0]; // inImage[pxIndex + (-1 + (-1 * width))]
      outIntensity += static_cast<double>(inImage[pxIndex - width_val]) * kernel[1];   // inImage[pxIndex + ( 0 + (-1 * width))]
      outIntensity += static_cast<double>(inImage[pxIndex - width_minus_1]) * kernel[2]; // inImage[pxIndex + ( 1 + (-1 * width))]

      // krow = 0
      outIntensity += static_cast<double>(inImage[pxIndex - 1]) * kernel[3];         // inImage[pxIndex + (-1 + ( 0 * width))]
      outIntensity += static_cast<double>(inImage[pxIndex]) * kernel[4];             // inImage[pxIndex + ( 0 + ( 0 * width))]
      outIntensity += static_cast<double>(inImage[pxIndex + 1]) * kernel[5];         // inImage[pxIndex + ( 1 + ( 0 * width))]

      // krow = OFFSET (equivalent to krow = 1 for a 3x3 kernel)
      outIntensity += static_cast<double>(inImage[pxIndex + width_minus_1]) * kernel[6]; // inImage[pxIndex + (-1 + ( 1 * width))]
      outIntensity += static_cast<double>(inImage[pxIndex + width_val]) * kernel[7];   // inImage[pxIndex + ( 0 + ( 1 * width))]
      outIntensity += static_cast<double>(inImage[pxIndex + width_plus_1]) * kernel[8]; // inImage[pxIndex + ( 1 + ( 1 * width))]

      // Cast the accumulated double intensity to uint8_t and store the result.
      outImage[pxIndex] = static_cast<uint8_t>(outIntensity);
    }
  }
}