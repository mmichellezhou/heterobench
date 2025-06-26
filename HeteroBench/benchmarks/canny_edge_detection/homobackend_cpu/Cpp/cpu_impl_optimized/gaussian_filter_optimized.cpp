#include "cpu_impl.h"
#include <cstring>
#include <iostream>
#include <math.h>

using namespace std;

void gaussian_filter_optimized(const uint8_t *inImage, int height, int width,
                               uint8_t *outImage) {
  gaussian_filter(inImage, height, width, outImage);
}