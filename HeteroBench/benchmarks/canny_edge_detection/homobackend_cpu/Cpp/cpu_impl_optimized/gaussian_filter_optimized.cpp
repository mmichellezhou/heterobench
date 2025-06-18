#include "cpu_impl.h"

void gaussian_filter_optimized(const uint8_t *inImage, int height, int width,
                               uint8_t *outImage) {
  gaussian_filter(inImage, height, width, outImage);
}