#include "cpu_impl.h"

void matmul_optimized(double *matmul_x, double *matmul_y, double *matmul_output, 
                int batch_size, int input_h, int input_w, int output_w) {
    matmul(matmul_x, matmul_y, matmul_output, batch_size, input_h, input_w, output_w);
}
