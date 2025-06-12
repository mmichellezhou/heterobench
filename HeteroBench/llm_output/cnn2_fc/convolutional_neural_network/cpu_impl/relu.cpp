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
#include <cmath> // Use cmath for fmax, which is typically more optimized for floating-point max

using namespace std;

/* This is the Cpp implementation of the following Python code */
/* Here the input and output are 2D arraies */
/*
def relu(x):
  return np.maximum(0, x)
*/

// Optimized relu implementation
//
// This implementation focuses on single-threaded performance improvements by:
// 1.  **Using `__restrict__` keyword:** This is a common compiler extension (supported by GCC, Clang, etc.)
//     that informs the compiler that the `relu_input` and `relu_output` pointers do not alias
//     (i.e., they point to distinct memory regions). This crucial hint allows the compiler
//     to perform more aggressive optimizations, particularly auto-vectorization (SIMD),
//     without needing to worry about potential data dependencies through aliased pointers.
// 2.  **Using `fmax` from `<cmath>`:** `fmax` is a C standard library function specifically designed
//     for floating-point maximum operations. It often maps directly to a single, highly optimized
//     Floating-Point Unit (FPU) instruction (e.g., `MAXPD` on x86-64 for doubles), which can be
//     faster than `std::max` from `<algorithm>` for floating-point types, especially when
//     auto-vectorized.
//
// These changes enable the compiler (when invoked with appropriate optimization flags like -O3
// and -march=native) to generate highly efficient vectorized code, processing multiple `double`
// elements in parallel using SIMD instructions (e.g., SSE, AVX, AVX2, AVX-512).
void relu(double * __restrict__ relu_input, double * __restrict__ relu_output, int size) {
  for (int i = 0; i < size; i++) {
    relu_output[i] = fmax(0.0, relu_input[i]);
  }
}