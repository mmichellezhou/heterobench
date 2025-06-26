#include "cpu_impl.h"

using namespace std;

void sobel_filter_x_optimized(const uint8_t *input_image, int height, int width, double *sobel_x) {
  sobel_filter_x(input_image, height, width, sobel_x);
}