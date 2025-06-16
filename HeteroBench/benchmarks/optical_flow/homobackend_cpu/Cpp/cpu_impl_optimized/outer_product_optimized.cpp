#include "cpu_impl.h"

void outer_product_optimized(gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
                   outer_t outer_product[MAX_HEIGHT][MAX_WIDTH])
{ 
  outer_product(gradient, outer_product);
}