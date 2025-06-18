#include "cpu_impl.h"

void edge_thinning_optimized(double *intensity, uint8_t *direction, int height,
                             int width, double *outImage) {
  edge_thinning(intensity, direction, height, width, outImage);
}