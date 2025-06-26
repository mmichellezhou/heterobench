#include "cpu_impl.h"
#include <iostream>
#include <math.h>

using namespace std;

void dot_add_optimized(double *dot_add_input_x, double *dot_add_input_W, double *dot_add_input_b, double *dot_add_output, int x_h, int x_w, int W_h, int W_w) {
    dot_add(dot_add_input_x, dot_add_input_W, dot_add_input_b, dot_add_output, x_h, x_w, W_h, W_w);
}