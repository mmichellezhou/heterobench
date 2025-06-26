#include "cpu_impl.h"

using namespace std;

void transpose_optimized(double *transpose_x, double *transpose_output, 
                int batch_size, int input_h, int input_w, int dim0, int dim1) {
    transpose(transpose_x, transpose_output, batch_size, input_h, input_w, dim0, dim1);
}