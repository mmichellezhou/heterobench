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
#include <cstring>    // Required for std::memcpy
#include <algorithm>  // Required for std::fill

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

void pad_input(double *pad_input_input, double *pad_input_output, int input_h, int input_w, int padding) {
  if (padding == 0) {
    // Case 1: No padding. Copy input directly to output.
    // Using std::memcpy for this contiguous block copy is highly optimized.
    // It leverages SIMD instructions and cache-friendly access patterns internally.
    // Use size_t for the number of elements to prevent potential integer overflow
    // when calculating total size for very large arrays.
    size_t num_elements = (size_t)input_h * input_w;
    std::memcpy(pad_input_output, pad_input_input, num_elements * sizeof(double));
    return;
  }

  // Case 2: Padding > 0.
  // Calculate the dimensions of the padded output array.
  int padded_h = input_h + 2 * padding;
  int padded_w = input_w + 2 * padding;

  // Step 1: Initialize the entire padded output array to 0.0.
  // std::fill is an efficient way to set a contiguous block of memory to a constant value.
  // Compilers are typically very good at optimizing and vectorizing std::fill for primitive types.
  // Use size_t for total elements to prevent potential integer overflow.
  size_t total_padded_elements = (size_t)padded_h * padded_w;
  std::fill(pad_input_output, pad_input_output + total_padded_elements, 0.0);

  // Step 2: Copy the original input data into the center region of the padded output array.
  // Optimize by pre-calculating row strides and using std::memcpy for each row.
  // This approach applies strength reduction (moving multiplications out of the inner loop)
  // and utilizes the highly optimized std::memcpy for contiguous row-wise copies.
  // std::memcpy is designed to be cache-efficient and uses SIMD instructions where possible.
  long long input_row_stride = input_w;
  long long output_row_stride = padded_w;

  for (int i = 0; i < input_h; i++) {
    // Calculate the starting address for the current row in both the input and output arrays.
    // Using long long for index calculations prevents potential integer overflow when dealing
    // with large dimensions, ensuring correct pointer arithmetic for large arrays.
    double* src_row_ptr = pad_input_input + (long long)i * input_row_stride;
    double* dest_row_ptr = pad_input_output + (long long)(i + padding) * output_row_stride + padding;

    // Copy the entire row from the source to the destination using std::memcpy.
    // This is generally much faster than a manual loop due to std::memcpy's
    // highly optimized implementation.
    std::memcpy(dest_row_ptr, src_row_ptr, (size_t)input_w * sizeof(double));
  }
}