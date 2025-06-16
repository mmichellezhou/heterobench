#include "cpu_impl.h"

void pad_input_optimized(double *pad_input_input, double *pad_input_output, int input_h, int input_w, int padding) {
    pad_input(pad_input_input, pad_input_output, input_h, input_w, padding);
}