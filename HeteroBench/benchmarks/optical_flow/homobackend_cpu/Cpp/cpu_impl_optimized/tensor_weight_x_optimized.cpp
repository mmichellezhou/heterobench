#include "cpu_impl.h"

void tensor_weight_x_optimized(tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor[MAX_HEIGHT][MAX_WIDTH])
{
  // TENSOR_FILTER is assumed to be a global/static const array, e.g., float TENSOR_FILTER[3];
  // tensor_t is assumed to be a struct like: struct tensor_t { float val[6]; };

  for (int r = 0; r < MAX_HEIGHT; r ++)
  {
    // Initialize a zero-valued tensor_t. This will be used for boundary conditions.
    // Using {} for aggregate initialization ensures all members are zero-initialized.
    tensor_t zero_tensor = {};

    // Handle the first output element tensor[r][0].
    // In the original code, for c=1, acc is initialized to zero, and then tensor[r][0] = acc.
    // This means tensor[r][0] is always zero.
    tensor[r][0] = zero_tensor;

    // Initialize the sliding window for memory access optimization.
    // These variables will hold tensor_y[r][c-2], tensor_y[r][c-1], and tensor_y[r][c] respectively.
    // For the first iteration of the main loop (c=2):
    // y_prev2 will be tensor_y[r][0]
    // y_prev1 will be tensor_y[r][1]
    tensor_t y_prev2 = tensor_y[r][0];
    tensor_t y_prev1 = tensor_y[r][1];

    // Main computation loop for columns.
    // The original 'c' loop ran from 0 to MAX_WIDTH.
    // We've handled c=0 and c=1 (writing to tensor[r][0]) separately.
    // The core computation in the original code happens when (c >= 2 && c < MAX_WIDTH).
    // This loop now directly covers that range, from c=2 up to MAX_WIDTH-1.
    // This eliminates the conditional branch inside the hot loop.
    for (int c = 2; c < MAX_WIDTH; c ++)
    {
      // Load the current tensor_y element for this iteration.
      // This completes the sliding window: y_prev2, y_prev1, y_curr.
      tensor_t y_curr = tensor_y[r][c];

      // Accumulator for the current output element, zero-initialized.
      tensor_t acc = {};

      // Fully unroll the 'i' loop (3 iterations) and the 'component' loop (6 iterations).
      // This reduces loop overhead, maximizes instruction-level parallelism,
      // and allows the compiler to keep accumulator components in registers.

      // i = 0: uses y_curr (tensor_y[r][c])
      const auto filter_val_0 = TENSOR_FILTER[0];
      acc.val[0] += y_curr.val[0] * filter_val_0;
      acc.val[1] += y_curr.val[1] * filter_val_0;
      acc.val[2] += y_curr.val[2] * filter_val_0;
      acc.val[3] += y_curr.val[3] * filter_val_0;
      acc.val[4] += y_curr.val[4] * filter_val_0;
      acc.val[5] += y_curr.val[5] * filter_val_0;

      // i = 1: uses y_prev1 (tensor_y[r][c-1])
      const auto filter_val_1 = TENSOR_FILTER[1];
      acc.val[0] += y_prev1.val[0] * filter_val_1;
      acc.val[1] += y_prev1.val[1] * filter_val_1;
      acc.val[2] += y_prev1.val[2] * filter_val_1;
      acc.val[3] += y_prev1.val[3] * filter_val_1;
      acc.val[4] += y_prev1.val[4] * filter_val_1;
      acc.val[5] += y_prev1.val[5] * filter_val_1;

      // i = 2: uses y_prev2 (tensor_y[r][c-2])
      const auto filter_val_2 = TENSOR_FILTER[2];
      acc.val[0] += y_prev2.val[0] * filter_val_2;
      acc.val[1] += y_prev2.val[1] * filter_val_2;
      acc.val[2] += y_prev2.val[2] * filter_val_2;
      acc.val[3] += y_prev2.val[3] * filter_val_2;
      acc.val[4] += y_prev2.val[4] * filter_val_2;
      acc.val[5] += y_prev2.val[5] * filter_val_2;

      // Write the accumulated value to the output tensor.
      // This corresponds to tensor[r][c-1] in the original code.
      tensor[r][c-1] = acc;

      // Update the sliding window for the next iteration.
      // The current y_prev1 becomes the new y_prev2.
      // The current y_curr becomes the new y_prev1.
      y_prev2 = y_prev1;
      y_prev1 = y_curr;
    }

    // Handle the last output element tensor[r][MAX_WIDTH-1].
    // In the original code, for c=MAX_WIDTH, acc is initialized to zero,
    // the computation block is skipped (because c < MAX_WIDTH is false),
    // and then tensor[r][MAX_WIDTH-1] = acc.
    // This means tensor[r][MAX_WIDTH-1] is always zero.
    tensor[r][MAX_WIDTH-1] = zero_tensor;
  }
}