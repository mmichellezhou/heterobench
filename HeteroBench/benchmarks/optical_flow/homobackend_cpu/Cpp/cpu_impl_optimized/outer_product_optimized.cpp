#include "cpu_impl.h"
#include <cstdio>

void outer_product_optimized(gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
                             outer_t outer_result[MAX_HEIGHT][MAX_WIDTH])
{
  outer_product(gradient, outer_result);
}