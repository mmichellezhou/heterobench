#include "cpu_impl.h"

void gradient_weight_x_optimized(gradient_t y_filt[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  gradient_weight_x(y_filt, filt_grad);
}