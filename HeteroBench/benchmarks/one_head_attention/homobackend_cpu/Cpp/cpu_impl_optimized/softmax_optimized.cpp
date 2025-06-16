#include "cpu_impl.h"

void get_max_optimized(double *get_max_x, double *get_max_max_x, 
                int batch_size, int input_h, int input_w, int axis, bool keepdims) {
    get_max(get_max_x, get_max_max_x, batch_size, input_h, input_w, axis, keepdims);
}

void get_sum_optimized(double *get_sum_x, double *get_sum_sum_x, 
                int batch_size, int input_h, int input_w, int axis, bool keepdims) {
    get_sum(get_sum_x, get_sum_sum_x, batch_size, input_h, input_w, axis, keepdims);
}

void get_exp_optimized(double *get_exp_x, double *get_exp_output, int batch_size, int input_h, int input_w) {
    get_exp(get_exp_x, get_exp_output, batch_size, input_h, input_w);
}

void softmax_optimized(double *softmax_x, double *softmax_output, 
             int batch_size, int input_h, int input_w, int axis) {
    softmax(softmax_x, softmax_output, batch_size, input_h, input_w, axis);
}