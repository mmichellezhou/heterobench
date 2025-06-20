#include "cpu_impl.h"

void pad_input_optimized(double *pad_input_input, double *pad_input_output, int input_h, int input_w, int padding) {
  // Handle the special case where no padding is required.
  // This involves a direct copy from input to output.
  if (padding == 0) {
    double* current_input_row_ptr = pad_input_input;
    double* current_output_row_ptr = pad_input_output;

    // Define an unroll factor for scalar operations.
    // Unrolling helps reduce loop overhead and expose instruction-level parallelism.
    const int UNROLL_FACTOR = 4; 

    for (int i = 0; i < input_h; i++) {
      int j = 0;
      // Process elements in chunks using unrolling
      for (; j + UNROLL_FACTOR <= input_w; j += UNROLL_FACTOR) {
        current_output_row_ptr[j] = current_input_row_ptr[j];
        current_output_row_ptr[j+1] = current_input_row_ptr[j+1];
        current_output_row_ptr[j+2] = current_input_row_ptr[j+2];
        current_output_row_ptr[j+3] = current_input_row_ptr[j+3];
      }
      // Handle remaining elements (if input_w is not a multiple of UNROLL_FACTOR)
      for (; j < input_w; j++) {
        current_output_row_ptr[j] = current_input_row_ptr[j];
      }
      // Move pointers to the next row
      current_input_row_ptr += input_w;
      current_output_row_ptr += input_w; // For padding == 0, output width is same as input width
    }
    return;
  }

  // Calculate the width of the padded output array.
  // This is a loop-invariant calculation moved outside the loops (loop-invariant code motion).
  int output_w = input_w + 2 * padding;

  // Define an unroll factor for scalar operations.
  const int UNROLL_FACTOR = 4;

  // Optimization Strategy for padding > 0:
  // Instead of zeroing the entire output array and then copying the input,
  // we perform a single pass that zeroes the padding regions and copies the input data,
  // avoiding redundant writes to memory. This improves cache efficiency and reduces memory traffic.

  // 1. Zero the top 'padding' rows of the output array.
  double* current_output_row_ptr = pad_input_output;
  for (int i = 0; i < padding; ++i) {
    int j = 0;
    // Unrolled loop for zeroing
    for (; j + UNROLL_FACTOR <= output_w; j += UNROLL_FACTOR) {
      current_output_row_ptr[j] = 0.0;
      current_output_row_ptr[j+1] = 0.0;
      current_output_row_ptr[j+2] = 0.0;
      current_output_row_ptr[j+3] = 0.0;
    }
    // Handle remainder
    for (; j < output_w; ++j) {
      current_output_row_ptr[j] = 0.0;
    }
    current_output_row_ptr += output_w; // Move to the next output row
  }

  // 2. Process the 'input_h' rows where the actual input data resides.
  // For each such row, zero the left padding, copy the input data, and zero the right padding.
  double* current_input_row_ptr = pad_input_input;
  for (int i = 0; i < input_h; ++i) {
    // Zero the 'padding' elements on the left side of the current output row.
    int j = 0;
    for (; j + UNROLL_FACTOR <= padding; j += UNROLL_FACTOR) {
      current_output_row_ptr[j] = 0.0;
      current_output_row_ptr[j+1] = 0.0;
      current_output_row_ptr[j+2] = 0.0;
      current_output_row_ptr[j+3] = 0.0;
    }
    for (; j < padding; ++j) {
      current_output_row_ptr[j] = 0.0;
    }

    // Copy 'input_w' elements from the current input row to the padded output row.
    // The destination starts after the left padding.
    j = 0; // Reset inner loop counter
    double* dest_ptr_for_copy = current_output_row_ptr + padding;
    for (; j + UNROLL_FACTOR <= input_w; j += UNROLL_FACTOR) {
      dest_ptr_for_copy[j] = current_input_row_ptr[j];
      dest_ptr_for_copy[j+1] = current_input_row_ptr[j+1];
      dest_ptr_for_copy[j+2] = current_input_row_ptr[j+2];
      dest_ptr_for_copy[j+3] = current_input_row_ptr[j+3];
    }
    for (; j < input_w; ++j) {
      dest_ptr_for_copy[j] = current_input_row_ptr[j];
    }

    // Zero the 'padding' elements on the right side of the current output row.
    // The destination starts after the copied input data.
    j = 0; // Reset inner loop counter
    double* dest_ptr_for_right_padding = current_output_row_ptr + padding + input_w;
    for (; j + UNROLL_FACTOR <= padding; j += UNROLL_FACTOR) {
      dest_ptr_for_right_padding[j] = 0.0;
      dest_ptr_for_right_padding[j+1] = 0.0;
      dest_ptr_for_right_padding[j+2] = 0.0;
      dest_ptr_for_right_padding[j+3] = 0.0;
    }
    for (; j < padding; ++j) {
      dest_ptr_for_right_padding[j] = 0.0;
    }

    // Move pointers to the next input and output rows.
    current_input_row_ptr += input_w;
    current_output_row_ptr += output_w;
  }

  // 3. Zero the bottom 'padding' rows of the output array.
  // 'current_output_row_ptr' is already positioned at the start of the first bottom padding row.
  for (int i = 0; i < padding; ++i) {
    int j = 0;
    // Unrolled loop for zeroing
    for (; j + UNROLL_FACTOR <= output_w; j += UNROLL_FACTOR) {
      current_output_row_ptr[j] = 0.0;
      current_output_row_ptr[j+1] = 0.0;
      current_output_row_ptr[j+2] = 0.0;
      current_output_row_ptr[j+3] = 0.0;
    }
    // Handle remainder
    for (; j < output_w; ++j) {
      current_output_row_ptr[j] = 0.0;
    }
    current_output_row_ptr += output_w; // Move to the next output row
  }
}