#include "cpu_impl.h"
#include "omp.h"
#include <cstring>
#include <iostream>
#include <math.h>

using namespace std;

void hysteresis_optimized(uint8_t *inImage, int height, int width,
                          uint8_t *outImage) {
  hysteresis(inImage, height, width, outImage);
}
