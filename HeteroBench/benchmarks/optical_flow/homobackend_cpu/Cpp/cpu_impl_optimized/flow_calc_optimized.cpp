#include "cpu_impl.h"

void flow_calc_optimized(tensor_t tensors[MAX_HEIGHT][MAX_WIDTH],
               velocity_t output[MAX_HEIGHT][MAX_WIDTH])
{
  flow_calc(tensors, output);
}

