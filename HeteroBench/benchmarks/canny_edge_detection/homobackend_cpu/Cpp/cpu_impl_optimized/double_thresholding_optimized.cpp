#include "cpu_impl.h"
#include "omp.h"
#include <cstring>
#include <iostream>
#include <math.h>

using namespace std;

void double_thresholding_optimized(double *suppressed_image, int height,
                                   int width, int high_threshold,
                                   int low_threshold, uint8_t *outImage) {
  double_thresholding(suppressed_image, height, width, high_threshold, low_threshold, outImage);
}
