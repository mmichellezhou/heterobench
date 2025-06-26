#include "cpu_impl.h"

using namespace std;

void sobel_filter_y_optimized(const uint8_t *input_image, int height, int width, double *sobel_y) {
  sobel_filter_y(input_image, height, width, sobel_y);
}