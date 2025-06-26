#include "cpu_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

void sigmoid_optimized(double *sigmoid_input, double *sigmoid_output, int size) {
    sigmoid(sigmoid_input, sigmoid_output, size);
}