#include "cpu_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

void relu_optimized(double *relu_input, double *relu_output, int size) {
  relu(relu_input, relu_output, size);
}