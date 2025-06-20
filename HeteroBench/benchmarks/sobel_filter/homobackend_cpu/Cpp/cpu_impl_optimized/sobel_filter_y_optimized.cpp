#include "cpu_impl.h"

void sobel_filter_y_optimized(const uint8_t *input_image, int height, int width, double *sobel_y) {
  // Use ptrdiff_t for indices to avoid potential overflow with very large width/height
  // and for better pointer arithmetic on 64-bit systems.
  // ptrdiff_t is a standard C++ type and is generally available in a C++ environment.
  const ptrdiff_t p_width = width;

  // Loop over rows, skipping border rows (row 0 and height-1) as the Sobel filter
  // requires a 3x3 neighborhood.
  for (int row = 1; row < height - 1; ++row) {
    // Pre-calculate row pointers for the current row's neighbors and the output row.
    // This is a form of strength reduction, avoiding repeated multiplications by 'width'
    // inside the inner column loop and improving memory access locality by using direct pointers.
    const uint8_t *p_row_minus_1 = input_image + (ptrdiff_t)(row - 1) * p_width;
    const uint8_t *p_row_plus_1  = input_image + (ptrdiff_t)(row + 1) * p_width;
    double *p_sobel_y_row        = sobel_y     + (ptrdiff_t)row * p_width;

    // Define loop bounds for the column loop.
    // The inner loop processes columns from 1 to width - 2 (inclusive).
    const int col_start = 1;
    const int col_end   = width - 2;

    // Unroll factor for the column loop to expose Instruction-Level Parallelism (ILP)
    // and reduce loop overhead. A factor of 4 is chosen as a common effective unroll amount.
    const int unroll_factor = 4;

    // Calculate the limit for the unrolled loop.
    // This ensures that we only process full blocks of 'unroll_factor' columns.
    const int num_iterations = col_end - col_start + 1;
    const int num_full_blocks = num_iterations / unroll_factor;
    const int col_unroll_limit = col_start + num_full_blocks * unroll_factor - 1;

    // Unrolled loop for processing columns in blocks of 'unroll_factor'.
    // The 3x3 convolution is fully unrolled and simplified by observing that
    // the middle row of the kernel_y is all zeros, thus its contribution is zero.
    // This eliminates two inner loops and 3 multiplications/additions per pixel.
    for (int col = col_start; col <= col_unroll_limit; col += unroll_factor) {
      // Calculate gy for current column 'col'
      double gy0 = 0;
      gy0 += (int)p_row_minus_1[col - 1] * (-1);
      gy0 += (int)p_row_minus_1[col    ] * (-2);
      gy0 += (int)p_row_minus_1[col + 1] * (-1);
      gy0 += (int)p_row_plus_1[col - 1] * ( 1);
      gy0 += (int)p_row_plus_1[col    ] * ( 2);
      gy0 += (int)p_row_plus_1[col + 1] * ( 1);
      p_sobel_y_row[col] = gy0;

      // Calculate gy for column 'col + 1'
      double gy1 = 0;
      gy1 += (int)p_row_minus_1[col    ] * (-1); // Corresponds to (col+1)-1
      gy1 += (int)p_row_minus_1[col + 1] * (-2); // Corresponds to (col+1)
      gy1 += (int)p_row_minus_1[col + 2] * (-1); // Corresponds to (col+1)+1
      gy1 += (int)p_row_plus_1[col    ] * ( 1);
      gy1 += (int)p_row_plus_1[col + 1] * ( 2);
      gy1 += (int)p_row_plus_1[col + 2] * ( 1);
      p_sobel_y_row[col + 1] = gy1;

      // Calculate gy for column 'col + 2'
      double gy2 = 0;
      gy2 += (int)p_row_minus_1[col + 1] * (-1);
      gy2 += (int)p_row_minus_1[col + 2] * (-2);
      gy2 += (int)p_row_minus_1[col + 3] * (-1);
      gy2 += (int)p_row_plus_1[col + 1] * ( 1);
      gy2 += (int)p_row_plus_1[col + 2] * ( 2);
      gy2 += (int)p_row_plus_1[col + 3] * ( 1);
      p_sobel_y_row[col + 2] = gy2;

      // Calculate gy for column 'col + 3'
      double gy3 = 0;
      gy3 += (int)p_row_minus_1[col + 2] * (-1);
      gy3 += (int)p_row_minus_1[col + 3] * (-2);
      gy3 += (int)p_row_minus_1[col + 4] * (-1);
      gy3 += (int)p_row_plus_1[col + 2] * ( 1);
      gy3 += (int)p_row_plus_1[col + 3] * ( 2);
      gy3 += (int)p_row_plus_1[col + 4] * ( 1);
      p_sobel_y_row[col + 3] = gy3;
    }

    // Cleanup loop for any remaining columns that didn't fit into full unrolled blocks.
    // This ensures all pixels are processed correctly, even for image widths not
    // perfectly divisible by the unroll factor.
    for (int col = col_unroll_limit + 1; col <= col_end; ++col) {
      double gy = 0;
      gy += (int)p_row_minus_1[col - 1] * (-1);
      gy += (int)p_row_minus_1[col    ] * (-2);
      gy += (int)p_row_minus_1[col + 1] * (-1);
      gy += (int)p_row_plus_1[col - 1] * ( 1);
      gy += (int)p_row_plus_1[col    ] * ( 2);
      gy += (int)p_row_plus_1[col + 1] * ( 1);
      p_sobel_y_row[col] = gy;
    }
  }
}