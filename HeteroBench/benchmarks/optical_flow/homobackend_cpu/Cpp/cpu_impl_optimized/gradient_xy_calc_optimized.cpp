#include "cpu_impl.h"

void gradient_xy_calc_optimized(pixel_t frame[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH])
{
  gradient_xy_calc(frame, gradient_x, gradient_y);
}