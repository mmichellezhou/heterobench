#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void transpose(double *transpose_x, double *transpose_output, int batch_size,
               int input_h, int input_w, int dim0, int dim1);
void matmul(double *matmul_x, double *matmul_y, double *matmul_output,
            int batch_size, int input_h, int input_w, int output_w);
void get_max(double *get_max_x, double *get_max_max_x, int batch_size,
             int input_h, int input_w, int axis, bool keepdims);
void get_sum(double *get_sum_x, double *get_sum_sum_x, int batch_size,
             int input_h, int input_w, int axis, bool keepdims);
void get_exp(double *get_exp_x, double *get_exp_output, int batch_size,
             int input_h, int input_w);
void softmax(double *softmax_x, double *softmax_output, int batch_size,
             int input_h, int input_w, int axis);