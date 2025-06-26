#include "cpu_impl.h"

FeatureType dotProduct_optimized(FeatureType param[NUM_FEATURES],
                                 DataType    feature[NUM_FEATURES])
{
  dotProduct(param, feature);
}
