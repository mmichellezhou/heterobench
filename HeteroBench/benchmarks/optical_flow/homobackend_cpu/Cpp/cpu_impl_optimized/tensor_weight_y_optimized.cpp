#include "cpu_impl.h"
#include <cstdio>

void tensor_weight_y_optimized(outer_t outer[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH])
{
  tensor_weight_y(outer, tensor_y);
}