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

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def pad_input(input, padding):
  if padding == 0:
    return input
  padded_input = np.zeros((input.shape[0] + 2*padding, input.shape[1] + 2*padding))
  for i in range(input.shape[0]):
    for j in range(input.shape[1]):
      padded_input[i + padding][j + padding] = input[i][j]
  return padded_input
*/

void pad_input(double * restrict pad_input_input, double * restrict pad_input_output, int input_h, int input_w, int padding) {
  if (padding == 0) {
    // Case 1: No padding. Copy input to output.
    // Optimization: Calculate row base pointers once per outer loop (strength reduction).
    // This helps the compiler generate more efficient code and facilitates auto-vectorization.
    // The 'restrict' keyword informs the compiler that input and output pointers do not alias,
    // enabling more aggressive optimizations.
    for (int i = 0; i < input_h; i++) {
      double* current_input_row_ptr = pad_input_input + i * input_w;
      double* current_output_row_ptr = pad_input_output + i * input_w;

      // Compiler pragmas to encourage auto-vectorization for the inner loop.
      // These are specific to GCC/Clang and hint that loop iterations are independent.
      #pragma GCC ivdep
      #pragma clang loop vectorize(enable)
      for (int j = 0; j < input_w; j++) {
        current_output_row_ptr[j] = current_input_row_ptr[j];
      }
    }
    return;
  }

  // Case 2: Padding > 0.
  int output_h = input_h + 2 * padding;
  int output_w = input_w + 2 * padding;

  // Step 1: Initialize the entire output array to 0.
  // Optimization: Calculate row base pointer once per outer loop (strength reduction).
  // This loop is highly amenable to SIMD vectorization.
  for (int i = 0; i < output_h; i++) {
    double* current_output_row_ptr = pad_input_output + i * output_w;

    // Compiler pragmas to encourage auto-vectorization.
    #pragma GCC ivdep
    #pragma clang loop vectorize(enable)
    for (int j = 0; j < output_w; j++) {
      current_output_row_ptr[j] = 0.0; // Use 0.0 for double literal
    }
  }

  // Step 2: Copy the input data to the padded central region of the output array.
  // Optimization: Calculate source and destination row base pointers once per outer loop.
  // The destination row offset includes the padding for rows and the starting column offset.
  // This loop is also highly amenable to SIMD vectorization.
  for (int i = 0; i < input_h; i++) {
    double* current_input_row_ptr = pad_input_input + i * input_w;
    // Calculate the starting address for the current row in the padded output.
    // (i + padding) accounts for row offset due to top padding.
    // + padding accounts for column offset due to left padding.
    double* current_output_row_ptr = pad_input_output + (i + padding) * output_w + padding;

    // Compiler pragmas to encourage auto-vectorization.
    #pragma GCC ivdep
    #pragma clang loop vectorize(enable)
    for (int j = 0; j < input_w; j++) {
      current_output_row_ptr[j] = current_input_row_ptr[j];
    }
  }
}