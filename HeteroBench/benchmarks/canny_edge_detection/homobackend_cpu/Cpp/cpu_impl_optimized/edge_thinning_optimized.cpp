#include "cpu_impl.h"
#include "omp.h"
#include <cstring>
#include <iostream>
#include <math.h>

using namespace std;

void edge_thinning_optimized(double *intensity, uint8_t *direction, int height,
                             int width, double *outImage) {
  edge_thinning(intensity, direction, height, width, outImage);
}