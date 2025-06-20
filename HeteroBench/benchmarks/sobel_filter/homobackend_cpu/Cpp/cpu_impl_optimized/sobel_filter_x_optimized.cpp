#include "cpu_impl.h"

void sobel_filter_x_optimized(const uint8_t *input_image, int height, int width, double *sobel_x) {
  // The kernel values are hardcoded into the calculations, eliminating the need for the array.
  // The kernel_x values are:
  // {-1, 0, 1},
  // {-2, 0, 2},
  // {-1, 0, 1}
  // The '0' columns mean those terms contribute nothing to the sum, so they are omitted.

  // Loop over rows, skipping the border rows (row 0 and height-1)
  for (int row = 1; row < height - 1; ++row) {
    // Pre-calculate row pointers to avoid repeated multiplications inside the inner loop.
    // These pointers provide direct access to the relevant rows of the input image.
    const uint8_t *p_m1 = input_image + (row - 1) * width; // Pointer to row-1
    const uint8_t *p_0  = input_image + row * width;       // Pointer to row
    const uint8_t *p_p1 = input_image + (row + 1) * width; // Pointer to row+1
    double *p_out = sobel_x + row * width;                 // Pointer to output row

    // Define unroll factor for the column loop.
    // Unrolling by 4 exposes more instruction-level parallelism and reduces loop overhead.
    const int UNROLL_FACTOR = 4;

    // Determine the loop bounds for the main unrolled loop and the remainder loop.
    // The original loop iterates from col = 1 up to width - 2 (inclusive).
    // col_limit is the exclusive upper bound for the original loop (width - 1).
    int col_limit = width - 1;
    int col_start = 1;
    // col_main_loop_end ensures that the unrolled loop processes a multiple of UNROLL_FACTOR
    // and leaves the remaining columns for the cleanup loop.
    int col_main_loop_end = col_limit - ((col_limit - col_start) % UNROLL_FACTOR);

    // Initialize sliding window variables for the first pixel (col=1).
    // We need three values from each of the three rows: (col-1), (col), and (col+1).
    // For col=1, these correspond to indices 0, 1, and 2.
    // Values are loaded as int to avoid repeated uint8_t to int promotions during arithmetic.
    int val_m1_prev = p_m1[0]; // Corresponds to p_m1[col-1] for col=1
    int val_m1_curr = p_m1[1]; // Corresponds to p_m1[col] for col=1
    int val_m1_next = p_m1[2]; // Corresponds to p_m1[col+1] for col=1

    int val_0_prev = p_0[0];
    int val_0_curr = p_0[1];
    int val_0_next = p_0[2];

    int val_p1_prev = p_p1[0];
    int val_p1_curr = p_p1[1];
    int val_p1_next = p_p1[2];

    // Main loop for columns, unrolled by UNROLL_FACTOR (4 iterations per loop pass).
    // This loop processes columns in blocks of 4.
    for (int col = col_start; col < col_main_loop_end; col += UNROLL_FACTOR) {
      // Calculate gx for the current 'col'
      // The calculation is expanded from the kernel:
      // gx = (-1 * P(r-1,c-1)) + (1 * P(r-1,c+1)) +
      //      (-2 * P(r,c-1))   + (2 * P(r,c+1))   +
      //      (-1 * P(r+1,c-1)) + (1 * P(r+1,c+1))
      // Intermediate sum is kept as int for strength reduction, then cast to double.
      int gx0_int = -val_m1_prev + val_m1_next - 2 * val_0_prev + 2 * val_0_next - val_p1_prev + val_p1_next;
      p_out[col] = static_cast<double>(gx0_int);

      // Shift window and load new values for the next iteration (col+1)
      // The 'prev' value becomes 'curr', 'curr' becomes 'next', and a new 'next' is loaded.
      val_m1_prev = val_m1_curr;
      val_m1_curr = val_m1_next;
      val_m1_next = p_m1[col + 2]; // Load p_m1[current_col_index + 1 + 1]

      val_0_prev = val_0_curr;
      val_0_curr = val_0_next;
      val_0_next = p_0[col + 2];

      val_p1_prev = val_p1_curr;
      val_p1_curr = val_p1_next;
      val_p1_next = p_p1[col + 2];

      // Calculate gx for col+1
      int gx1_int = -val_m1_prev + val_m1_next - 2 * val_0_prev + 2 * val_0_next - val_p1_prev + val_p1_next;
      p_out[col + 1] = static_cast<double>(gx1_int);

      // Shift window and load new values for col+2
      val_m1_prev = val_m1_curr;
      val_m1_curr = val_m1_next;
      val_m1_next = p_m1[col + 3]; // Load p_m1[current_col_index + 2 + 1]

      val_0_prev = val_0_curr;
      val_0_curr = val_0_next;
      val_0_next = p_0[col + 3];

      val_p1_prev = val_p1_curr;
      val_p1_curr = val_p1_next;
      val_p1_next = p_p1[col + 3];

      // Calculate gx for col+2
      int gx2_int = -val_m1_prev + val_m1_next - 2 * val_0_prev + 2 * val_0_next - val_p1_prev + val_p1_next;
      p_out[col + 2] = static_cast<double>(gx2_int);

      // Shift window and load new values for col+3
      val_m1_prev = val_m1_curr;
      val_m1_curr = val_m1_next;
      val_m1_next = p_m1[col + 4]; // Load p_m1[current_col_index + 3 + 1]

      val_0_prev = val_0_curr;
      val_0_curr = val_0_next;
      val_0_next = p_0[col + 4];

      val_p1_prev = val_p1_curr;
      val_p1_curr = val_p1_next;
      val_p1_next = p_p1[col + 4];

      // Calculate gx for col+3
      int gx3_int = -val_m1_prev + val_m1_next - 2 * val_0_prev + 2 * val_0_next - val_p1_prev + val_p1_next;
      p_out[col + 3] = static_cast<double>(gx3_int);
    }

    // Remainder loop for columns not covered by the unrolled loop.
    // This loop handles the last few columns if (width - 2) is not a multiple of UNROLL_FACTOR.
    // Direct indexing is used here for simplicity and correctness, as the sliding window variables
    // might not be correctly aligned for the start of this loop.
    for (int col = col_main_loop_end; col < col_limit; ++col) {
      // Load pixel values directly for the current column and its neighbors.
      int pixel_val_m1_m1 = p_m1[col - 1];
      int pixel_val_m1_p1 = p_m1[col + 1];
      int pixel_val_0_m1  = p_0[col - 1];
      int pixel_val_0_p1  = p_0[col + 1];
      int pixel_val_p1_m1 = p_p1[col - 1];
      int pixel_val_p1_p1 = p_p1[col + 1];

      // Calculate gx using integer arithmetic for accumulation.
      int gx_int = -pixel_val_m1_m1 + pixel_val_m1_p1 - 2 * pixel_val_0_m1 + 2 * pixel_val_0_p1 - pixel_val_p1_m1 + pixel_val_p1_p1;
      // Store the result after casting to double.
      p_out[col] = static_cast<double>(gx_int);
    }
  }
}