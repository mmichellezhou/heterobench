#include "cpu_impl.h"

void relu_optimized(double *relu_input, double *relu_output, int size) {
  // Unrolling the loop to reduce loop overhead (branching, index increment)
  // and expose Instruction-Level Parallelism (ILP). This allows the CPU to
  // schedule multiple independent load, compute (max), and store operations
  // concurrently, making better use of execution units.
  // An unroll factor of 8 is chosen as a common balance for scalar double
  // operations on modern x86-64 architectures, considering the number of
  // available execution ports for loads, stores, and floating-point operations.
  const int unroll_factor = 8;
  int i = 0;

  // Process elements in chunks of 'unroll_factor'.
  // This loop handles the main bulk of the data, where 'size' is large enough.
  for (; i + (unroll_factor - 1) < size; i += unroll_factor) {
    // Each of these operations is independent of the others within the same
    // unrolled iteration, allowing the CPU's out-of-order execution engine
    // to pipeline them efficiently.
    relu_output[i] = std::max(0.0, relu_input[i]);
    relu_output[i+1] = std::max(0.0, relu_input[i+1]);
    relu_output[i+2] = std::max(0.0, relu_input[i+2]);
    relu_output[i+3] = std::max(0.0, relu_input[i+3]);
    relu_output[i+4] = std::max(0.0, relu_input[i+4]);
    relu_output[i+5] = std::max(0.0, relu_input[i+5]);
    relu_output[i+6] = std::max(0.0, relu_input[i+6]);
    relu_output[i+7] = std::max(0.0, relu_input[i+7]);
  }

  // Handle any remaining elements that did not fit into the unrolled loop.
  // This ensures functional equivalence for all possible 'size' values.
  for (; i < size; i++) {
    relu_output[i] = std::max(0.0, relu_input[i]);
  }
}