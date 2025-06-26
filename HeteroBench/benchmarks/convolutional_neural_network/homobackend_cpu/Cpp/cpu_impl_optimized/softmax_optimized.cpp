#include "cpu_impl.h"

using namespace std;

void softmax_optimized(double *softmax_input, double *exp_results, double *softmax_output, int size)
{
  softmax(softmax_input, exp_results, softmax_output, size);
}