#include "cpu_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

void max_pooling_optimized(double *max_pooling_input, int pool_size, int pool_stride, int input_h, int input_w, double *max_pooling_output) {
  max_pooling(max_pooling_input, pool_size, pool_stride, input_h, input_w, max_pooling_output);
}