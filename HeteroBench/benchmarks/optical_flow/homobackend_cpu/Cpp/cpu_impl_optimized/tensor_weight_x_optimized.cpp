#include "cpu_impl.h"

void tensor_weight_x_optimized(tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor[MAX_HEIGHT][MAX_WIDTH])
{
  tensor_weight_x(tensor_y, tensor);
}