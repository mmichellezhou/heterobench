#include "cpu_impl.h"

void updateParameter_optimized(
    FeatureType param[NUM_FEATURES],
    FeatureType grad[NUM_FEATURES],
    FeatureType step_size)
{
  updateParameter(param, grad, step_size);
}
