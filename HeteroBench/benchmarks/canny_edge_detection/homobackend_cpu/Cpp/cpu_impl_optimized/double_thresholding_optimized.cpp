#include "cpu_impl.h"

using namespace std;

void double_thresholding_optimized(double *suppressed_image, int height,
                                   int width, int high_threshold,
                                   int low_threshold, uint8_t *outImage) {
  double_thresholding(suppressed_image, height, width, high_threshold, low_threshold, outImage);
}
