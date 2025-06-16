#include "cpu_impl.h"

void hysteresis_optimized(uint8_t *inImage, int height, int width,
                uint8_t *outImage) {
    hysteresis(inImage, height, width, outImage);
}
