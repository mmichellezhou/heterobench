#include "cpu_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

void softmax_optimized(double *softmax_x, double *softmax_output, 
             int batch_size, int input_h, int input_w, int axis) {
    softmax(softmax_x, softmax_output, batch_size, input_h, input_w, axis);
}