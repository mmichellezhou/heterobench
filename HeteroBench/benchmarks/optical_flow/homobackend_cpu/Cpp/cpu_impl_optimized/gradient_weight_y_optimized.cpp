#include "cpu_impl.h"

void gradient_weight_y_optimized(pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
                       pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH],
                       pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH])
{
  gradient_weight_y(gradient_x, gradient_y, gradient_z, filt_grad);
}