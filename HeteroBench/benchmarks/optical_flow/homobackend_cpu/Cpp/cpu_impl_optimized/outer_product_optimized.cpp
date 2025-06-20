#include "cpu_impl.h"

void outer_product_optimized(gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
                             outer_t outer_result[MAX_HEIGHT][MAX_WIDTH])
{
  // Define a constant for the unroll factor. This constant is local to the function
  // and does not need to be defined outside.
  // Unrolling by 4 reduces loop overhead and exposes more instruction-level parallelism
  // for the CPU's out-of-order execution engine, without excessive register pressure.
  const int UNROLL_FACTOR = 4;

  for (int r = 0; r < MAX_HEIGHT; r++)
  {
    int c = 0;
    // Main loop: Process elements in chunks of UNROLL_FACTOR
    // This reduces the number of loop iterations, branch predictions, and
    // allows the CPU to schedule multiple independent computations concurrently.
    for (; c + UNROLL_FACTOR <= MAX_WIDTH; c += UNROLL_FACTOR)
    {
      // Process element at index c
      gradient_t grad0 = gradient[r][c];
      outer_t out0;
      out0.val[0] = grad0.x * grad0.x;
      out0.val[1] = grad0.y * grad0.y;
      out0.val[2] = grad0.z * grad0.z;
      out0.val[3] = grad0.x * grad0.y;
      out0.val[4] = grad0.x * grad0.z;
      out0.val[5] = grad0.y * grad0.z;
      outer_result[r][c] = out0;

      // Process element at index c+1
      gradient_t grad1 = gradient[r][c+1];
      outer_t out1;
      out1.val[0] = grad1.x * grad1.x;
      out1.val[1] = grad1.y * grad1.y;
      out1.val[2] = grad1.z * grad1.z;
      out1.val[3] = grad1.x * grad1.y;
      out1.val[4] = grad1.x * grad1.z;
      out1.val[5] = grad1.y * grad1.z;
      outer_result[r][c+1] = out1;

      // Process element at index c+2
      gradient_t grad2 = gradient[r][c+2];
      outer_t out2;
      out2.val[0] = grad2.x * grad2.x;
      out2.val[1] = grad2.y * grad2.y;
      out2.val[2] = grad2.z * grad2.z;
      out2.val[3] = grad2.x * grad2.y;
      out2.val[4] = grad2.x * grad2.z;
      out2.val[5] = grad2.y * grad2.z;
      outer_result[r][c+2] = out2;

      // Process element at index c+3
      gradient_t grad3 = gradient[r][c+3];
      outer_t out3;
      out3.val[0] = grad3.x * grad3.x;
      out3.val[1] = grad3.y * grad3.y;
      out3.val[2] = grad3.z * grad3.z;
      out3.val[3] = grad3.x * grad3.y;
      out3.val[4] = grad3.x * grad3.z;
      out3.val[5] = grad3.y * grad3.z;
      outer_result[r][c+3] = out3;
    }

    // Tail loop: Process any remaining elements that didn't fit into the unrolled chunks.
    // This ensures functional equivalence for MAX_WIDTH values not perfectly divisible
    // by UNROLL_FACTOR.
    for (; c < MAX_WIDTH; c++)
    {
      gradient_t grad = gradient[r][c];
      outer_t out;
      out.val[0] = grad.x * grad.x;
      out.val[1] = grad.y * grad.y;
      out.val[2] = grad.z * grad.z;
      out.val[3] = grad.x * grad.y;
      out.val[4] = grad.x * grad.z;
      out.val[5] = grad.y * grad.z;
      outer_result[r][c] = out;
    }
  }
}