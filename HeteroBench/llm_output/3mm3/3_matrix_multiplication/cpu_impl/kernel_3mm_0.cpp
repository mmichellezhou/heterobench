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
#include <immintrin.h> // Required for AVX2 intrinsics (for _mm256_ functions)

// The original code used 'using namespace std;', but it's not strictly necessary here
// and generally better to avoid in header files or large scopes.

void kernel_3m_0(double A[NI + 0][NK + 0], double B[NK + 0][NJ + 0], double E[NI + 0][NJ + 0])
{
  // Original loop order: i (c1), j (c2), k (c5)
  // E[c1][c2] += A[c1][c5] * B[c5][c2];
  // This original order results in non-contiguous memory access for B[c5][c2]
  // in the innermost loop (as c5 changes while c2 is fixed), leading to poor cache performance.

  // Optimized loop order: i (c1), k (c5), j (c2)
  // This order ensures contiguous memory access for E[c1][c2] and B[c5][c2]
  // in the innermost loop (c2). This significantly improves cache locality and
  // enables efficient vectorization (SIMD) of the innermost loop.
  // A[c1][c5] is loaded once per iteration of the middle loop (c5), which is also efficient.

  // Vectorization factor for double-precision floating-point numbers using AVX2 (256-bit registers).
  // A __m256d register can hold 4 doubles (4 * 8 bytes = 32 bytes).
  const int VEC_SIZE = 4; 

  // Loop over rows of A and E
  for (int c1 = 0; c1 < NI; c1++) { 
    // Loop over columns of A and rows of B
    for (int c5 = 0; c5 < NK; c5++) { 
      // Load A[c1][c5] once outside the innermost loop.
      // This value will be broadcasted to a vector register for SIMD multiplication.
      double val_A_c1_c5 = A[c1][c5];
      __m256d vec_val_A = _mm256_set1_pd(val_A_c1_c5); // Broadcast scalar to all elements of a vector

      // Vectorized loop for the innermost dimension (c2 - NJ)
      // Process 'NJ' elements in chunks of 'VEC_SIZE'.
      // The loop condition `c2 < NJ / VEC_SIZE * VEC_SIZE` ensures we only process full vectors.
      for (int c2 = 0; c2 < NJ / VEC_SIZE * VEC_SIZE; c2 += VEC_SIZE) {
        // Load 4 doubles from E. _mm256_loadu_pd is used for unaligned loads,
        // which is safe and generally preferred for function parameters where
        // alignment cannot be guaranteed by the caller.
        __m256d vec_E = _mm256_loadu_pd(&E[c1][c2]);
        // Load 4 doubles from B (unaligned load)
        __m256d vec_B = _mm256_loadu_pd(&B[c5][c2]);

        // Perform SIMD multiplication: val_A * B
        __m256d prod = _mm256_mul_pd(vec_val_A, vec_B);

        // Perform SIMD addition: E + prod
        __m256d sum = _mm256_add_pd(vec_E, prod);

        // Store the result back to E (unaligned store)
        _mm256_storeu_pd(&E[c1][c2], sum);
      }

      // Scalar epilogue loop for remaining elements (if NJ is not a multiple of VEC_SIZE).
      // This ensures correctness for all NJ sizes.
      for (int c2 = NJ / VEC_SIZE * VEC_SIZE; c2 < NJ; c2++) {
        E[c1][c2] += val_A_c1_c5 * B[c5][c2];
      }
    }
  }
}