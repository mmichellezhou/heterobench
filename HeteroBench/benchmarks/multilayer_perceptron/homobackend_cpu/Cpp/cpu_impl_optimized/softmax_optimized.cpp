#include "cpu_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

void softmax_optimized(double *softmax_input, double *exp_results, double *softmax_output, int size) {
    softmax(softmax_input, exp_results, softmax_output, size);
}