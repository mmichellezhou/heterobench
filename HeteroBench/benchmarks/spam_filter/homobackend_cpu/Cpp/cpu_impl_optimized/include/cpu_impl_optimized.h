#ifndef __CPU_IMPL_OPTIMIZED_H__
#define __CPU_IMPL_OPTIMIZED_H__

#include "math.h"

// dataset information
// const int NUM_FEATURES  = 1024;
// const int NUM_SAMPLES   = 5000;
// const int NUM_TRAINING  = 4500;
// const int NUM_TESTING   = 500;
// const int STEP_SIZE     = 60000;
// const int NUM_EPOCHS    = 5;
// const int DATA_SET_SIZE = NUM_FEATURES * NUM_SAMPLES;

FeatureType dotProduct_optimized(FeatureType param[NUM_FEATURES],
                                 DataType feature[NUM_FEATURES]);

FeatureType Sigmoid_optimized(FeatureType exponent);

void computeGradient_optimized(FeatureType grad[NUM_FEATURES],
                               DataType feature[NUM_FEATURES],
                               FeatureType scale);

void updateParameter_optimized(FeatureType param[NUM_FEATURES],
                               FeatureType grad[NUM_FEATURES],
                               FeatureType step_size);

#endif
