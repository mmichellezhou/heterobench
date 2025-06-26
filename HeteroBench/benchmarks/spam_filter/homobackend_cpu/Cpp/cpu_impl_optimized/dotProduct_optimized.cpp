#include "cpu_impl.h"

FeatureType dotProduct_optimized(FeatureType param[NUM_FEATURES],
                                 DataType    feature[NUM_FEATURES])
{
  // Use multiple accumulators to expose Instruction-Level Parallelism (ILP).
  // This allows the CPU to perform multiple independent multiply-add operations
  // in parallel, reducing the critical path latency of the reduction.
  FeatureType result0 = 0;
  FeatureType result1 = 0;
  FeatureType result2 = 0;
  FeatureType result3 = 0;
  FeatureType result4 = 0;
  FeatureType result5 = 0;
  FeatureType result6 = 0;
  FeatureType result7 = 0;

  int i = 0;
  // Unroll the loop by 8 to reduce loop overhead (branching, index increments)
  // and to provide more independent operations for the CPU's execution units.
  // This helps in hiding the latency of floating-point operations and memory loads.
  for (; i + 7 < NUM_FEATURES; i += 8) {
    result0 += param[i] * feature[i];
    result1 += param[i+1] * feature[i+1];
    result2 += param[i+2] * feature[i+2];
    result3 += param[i+3] * feature[i+3];
    result4 += param[i+4] * feature[i+4];
    result5 += param[i+5] * feature[i+5];
    result6 += param[i+6] * feature[i+6];
    result7 += param[i+7] * feature[i+7];
  }

  // Handle remaining elements (tail loop) if NUM_FEATURES is not a multiple of 8.
  // This ensures functional equivalence for any NUM_FEATURES value.
  FeatureType final_result = 0;
  for (; i < NUM_FEATURES; i++) {
    final_result += param[i] * feature[i];
  }

  // Sum up the partial results from the multiple accumulators.
  // This final reduction step combines the parallel computations.
  final_result += result0 + result1 + result2 + result3 + result4 + result5 + result6 + result7;

  return final_result;
}
