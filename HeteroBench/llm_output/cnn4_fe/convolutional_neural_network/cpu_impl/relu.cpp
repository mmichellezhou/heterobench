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
#include <math.h> // For std::max in the scalar tail processing
#include <immintrin.h> // For AVX intrinsics (__m256d, _mm256_loadu_pd, _mm256_setzero_pd, _mm256_max_pd, _mm256_storeu_pd)

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def relu(x):
  return np.maximum(0, x)
*/

void relu(double *relu_input, double *relu_output, int size) {
  // Optimize using AVX (Advanced Vector Extensions) intrinsics for double-precision floating-point numbers.
  // An __m256d register can hold 4 double-precision floating-point values.
  const int VEC_SIZE = 4; // Number of doubles processed per AVX instruction

  int i = 0;
  // Process the array in chunks of VEC_SIZE using AVX instructions.
  // This loop handles the main part of the array that can be vectorized.
  for (; i + VEC_SIZE <= size; i += VEC_SIZE) {
    // Load 4 double-precision floating-point values from relu_input into an AVX register.
    // _mm256_loadu_pd is used for unaligned memory access, which is generally safer
    // as array alignment is not guaranteed by the function signature.
    __m256d input_vec = _mm256_loadu_pd(&relu_input[i]);

    // Create an AVX register filled with 0.0. This will be used as the second operand for max.
    __m256d zero_vec = _mm256_setzero_pd();

    // Compute the element-wise maximum between the input vector and the zero vector.
    // This effectively performs the ReLU operation (max(0.0, x)) on 4 doubles simultaneously.
    __m256d result_vec = _mm256_max_pd(input_vec, zero_vec);

    // Store the 4 resulting double-precision floating-point values back into relu_output.
    // _mm256_storeu_pd is used for unaligned memory access.
    _mm256_storeu_pd(&relu_output[i], result_vec);
  }

  // Handle any remaining elements (the "tail" of the array) that could not be processed
  // by the vectorized loop. This occurs if 'size' is not a multiple of VEC_SIZE.
  // These elements are processed using standard scalar operations.
  for (; i < size; i++) {
    relu_output[i] = max(0.0, relu_input[i]);
  }
}