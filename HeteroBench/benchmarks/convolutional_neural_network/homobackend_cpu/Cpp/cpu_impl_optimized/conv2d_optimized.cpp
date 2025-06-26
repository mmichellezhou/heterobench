#include "cpu_impl.h"

using namespace std;

void conv2d_optimized(double *conv2d_input, double *conv2d_kernel, double *input_padded, double conv2d_bias, int stride, int padding, int input_h, int input_w, int kernel_h, int kernel_w, double *conv2d_output)
{
  conv2d(conv2d_input, conv2d_kernel, input_padded, conv2d_bias, stride, padding, input_h, input_w, kernel_h, kernel_w, conv2d_output);
}