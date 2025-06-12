/*
 * (C) Copyright [2024] Hewlett Packard Enterprise Development LP
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
 
#include "cpu_impl.h"
#include <iostream>
#include <math.h>
#include <immintrin.h> // Required for AVX intrinsics

// The original code used 'using namespace std;', but it's generally better
// to explicitly qualify standard library functions like std::max.

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def relu(x):
  return np.maximum(0, x)
*/

void relu(double *relu_input, double *relu_output, int size) {
  // For optimal single-threaded performance, we leverage SIMD (Single Instruction, Multiple Data)
  // instructions, specifically AVX (Advanced Vector Extensions) for double-precision floating-point numbers.
  // AVX registers (__m256d) can hold 4 double-precision numbers.

  // Create a vector where all 4 double elements are 0.0. This will be used for the max(0.0, x) operation.
  const __m256d zero_vec = _mm256_set1_pd(0.0);

  // Process elements in chunks of 4 using AVX intrinsics.
  // This loop handles the main part of the array that can be processed efficiently with SIMD.
  int i = 0;
  // The loop condition `i + 3 < size` ensures that there are at least 4 elements remaining
  // to be processed by the AVX instructions (i.e., elements at indices i, i+1, i+2, i+3).
  for (; i + 3 < size; i += 4) {
    // Load 4 double-precision floating-point numbers from the input array starting at relu_input[i].
    // _mm256_loadu_pd performs an unaligned load, which is safe and correct for any memory address,
    // avoiding potential crashes or incorrect behavior due to memory alignment issues.
    __m256d input_vec = _mm256_loadu_pd(&relu_input[i]);

    // Compute the maximum of each element in 'input_vec' and the corresponding element in 'zero_vec'.
    // This effectively performs the ReLU operation (max(0.0, x)) on 4 doubles simultaneously.
    __m256d result_vec = _mm256_max_pd(zero_vec, input_vec);

    // Store the 4 resulting double-precision numbers back into the output array starting at relu_output[i].
    // _mm256_storeu_pd performs an unaligned store, ensuring correctness regardless of alignment.
    _mm256_storeu_pd(&relu_output[i], result_vec);
  }

  // Process any remaining elements (the "tail" of the array) using scalar operations.
  // This loop handles the elements that couldn't be processed in full 4-element chunks by the SIMD loop.
  for (; i < size; i++) {
    relu_output[i] = std::max(0.0, relu_input[i]);
  }
}