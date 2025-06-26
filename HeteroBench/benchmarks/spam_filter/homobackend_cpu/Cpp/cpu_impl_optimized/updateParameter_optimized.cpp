#include "cpu_impl.h"

void updateParameter_optimized(
    FeatureType param[NUM_FEATURES],
    FeatureType grad[NUM_FEATURES],
    FeatureType step_size)
{
  // Loop unrolling is applied to expose instruction-level parallelism (ILP)
  // and reduce loop overhead (branching, index increment).
  // By processing multiple elements within a single loop iteration,
  // the CPU can schedule independent floating-point operations concurrently
  // on its multiple execution units.
  // A common unroll factor like 8 is chosen to balance code size and ILP.

  int i = 0;

  // Unrolled loop for the main part of the array
  // Process 8 elements per iteration
  for (; i + 7 < NUM_FEATURES; i += 8) {
    // Each of these operations is independent and can potentially be
    // executed in parallel by the CPU's execution units.
    // The 'step_size' value will likely be held in a register by the compiler.
    param[i]     += step_size * grad[i];
    param[i + 1] += step_size * grad[i + 1];
    param[i + 2] += step_size * grad[i + 2];
    param[i + 3] += step_size * grad[i + 3];
    param[i + 4] += step_size * grad[i + 4];
    param[i + 5] += step_size * grad[i + 5];
    param[i + 6] += step_size * grad[i + 6];
    param[i + 7] += step_size * grad[i + 7];
  }

  // Remainder loop to handle any elements not covered by the unrolled loop.
  // This ensures functional equivalence for any NUM_FEATURES value.
  for (; i < NUM_FEATURES; i++) {
    param[i] += step_size * grad[i];
  }
}
