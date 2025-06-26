#include "cpu_impl.h"
#include "math.h"

void computeGradient_optimized(
    FeatureType grad[NUM_FEATURES],
    DataType    feature[NUM_FEATURES],
    FeatureType scale)
{
  computeGradient(grad, feature, scale);
}
