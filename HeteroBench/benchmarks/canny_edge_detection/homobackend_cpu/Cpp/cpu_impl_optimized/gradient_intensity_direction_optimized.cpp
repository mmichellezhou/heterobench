#include "cpu_impl.h"

using namespace std;

void gradient_intensity_direction_optimized(const uint8_t *inImage, int height,
                                            int width, double *intensity,
                                            uint8_t *direction) {
  gradient_intensity_direction(inImage, height, width, intensity, direction);
}