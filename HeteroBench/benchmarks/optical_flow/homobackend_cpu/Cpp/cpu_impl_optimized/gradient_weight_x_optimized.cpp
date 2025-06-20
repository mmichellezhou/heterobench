#include "cpu_impl.h"

void gradient_weight_x_optimized(gradient_t y_filt[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  for (int r = 0; r < MAX_HEIGHT; r ++)
  {
    // Part 1: Left border region (output indices k = 0, 1, 2)
    // In the original code, this corresponds to 'c' values of 3, 4, 5.
    // For these 'c' values, the condition 'if (c >= 6 && c < MAX_WIDTH)' is false,
    // and 'else if (c >= 3)' is true. This results in 'filt_grad[r][c-3]' being
    // assigned 'acc', which is initialized to zero.
    for (int k = 0; k < 3; k++) {
      filt_grad[r][k].x = 0;
      filt_grad[r][k].y = 0;
      filt_grad[r][k].z = 0;
    }

    // Part 2: Main computation region (output indices k = 3 to MAX_WIDTH - 4)
    // In the original code, this corresponds to 'c' values from 6 to MAX_WIDTH - 1.
    // For these 'c' values, the condition 'if (c >= 6 && c < MAX_WIDTH)' is true,
    // and the 7-iteration convolution loop is executed.
    // The loop condition 'k < MAX_WIDTH - 3' correctly handles cases where MAX_WIDTH is small
    // (e.g., MAX_WIDTH < 6), ensuring this loop does not execute if the range is invalid,
    // maintaining functional equivalence with the original conditional logic.
    for (int k = 3; k < MAX_WIDTH - 3; k ++)
    {
      // Use local accumulators to encourage the compiler to keep these values
      // in CPU registers throughout the computation, reducing memory traffic.
      // The type of acc_x, acc_y, acc_z is deduced from the members of gradient_t
      // to ensure type correctness (e.g., float or double).
      decltype(y_filt[0][0].x) acc_x = 0;
      decltype(y_filt[0][0].y) acc_y = 0;
      decltype(y_filt[0][0].z) acc_z = 0;

      // The inner convolution loop (for 'i' from 0 to 6) is fully unrolled.
      // This eliminates loop overhead (branching, counter increments) and exposes
      // maximum instruction-level parallelism (ILP) to the CPU's out-of-order execution engine.
      // The original index was y_filt[r][c-i], where 'c' is equivalent to 'k+3' in this transformed loop.
      // So, the access pattern becomes y_filt[r][(k+3)-i].
      
      // i = 0: y_filt[r][k+3] * GRAD_FILTER[0]
      acc_x += y_filt[r][k+3].x * GRAD_FILTER[0];
      acc_y += y_filt[r][k+3].y * GRAD_FILTER[0];
      acc_z += y_filt[r][k+3].z * GRAD_FILTER[0];

      // i = 1: y_filt[r][k+2] * GRAD_FILTER[1]
      acc_x += y_filt[r][k+2].x * GRAD_FILTER[1];
      acc_y += y_filt[r][k+2].y * GRAD_FILTER[1];
      acc_z += y_filt[r][k+2].z * GRAD_FILTER[1];

      // i = 2: y_filt[r][k+1] * GRAD_FILTER[2]
      acc_x += y_filt[r][k+1].x * GRAD_FILTER[2];
      acc_y += y_filt[r][k+1].y * GRAD_FILTER[2];
      acc_z += y_filt[r][k+1].z * GRAD_FILTER[2];

      // i = 3: y_filt[r][k] * GRAD_FILTER[3]
      acc_x += y_filt[r][k].x * GRAD_FILTER[3];
      acc_y += y_filt[r][k].y * GRAD_FILTER[3];
      acc_z += y_filt[r][k].z * GRAD_FILTER[3];

      // i = 4: y_filt[r][k-1] * GRAD_FILTER[4]
      acc_x += y_filt[r][k-1].x * GRAD_FILTER[4];
      acc_y += y_filt[r][k-1].y * GRAD_FILTER[4];
      acc_z += y_filt[r][k-1].z * GRAD_FILTER[4];

      // i = 5: y_filt[r][k-2] * GRAD_FILTER[5]
      acc_x += y_filt[r][k-2].x * GRAD_FILTER[5];
      acc_y += y_filt[r][k-2].y * GRAD_FILTER[5];
      acc_z += y_filt[r][k-2].z * GRAD_FILTER[5];

      // i = 6: y_filt[r][k-3] * GRAD_FILTER[6]
      acc_x += y_filt[r][k-3].x * GRAD_FILTER[6];
      acc_y += y_filt[r][k-3].y * GRAD_FILTER[6];
      acc_z += y_filt[r][k-3].z * GRAD_FILTER[6];

      // Store the accumulated result to the output array.
      filt_grad[r][k].x = acc_x;
      filt_grad[r][k].y = acc_y;
      filt_grad[r][k].z = acc_z;
    }

    // Part 3: Right border region (output indices k = MAX_WIDTH - 3 to MAX_WIDTH - 1)
    // In the original code, this corresponds to 'c' values from MAX_WIDTH to MAX_WIDTH + 2.
    // For these 'c' values, the condition 'if (c >= 6 && c < MAX_WIDTH)' is false,
    // and 'else if (c >= 3)' is true. This results in 'filt_grad[r][c-3]' being
    // assigned 'acc', which is initialized to zero.
    for (int k = MAX_WIDTH - 3; k < MAX_WIDTH; k++) {
      filt_grad[r][k].x = 0;
      filt_grad[r][k].y = 0;
      filt_grad[r][k].z = 0;
    }
  }
}