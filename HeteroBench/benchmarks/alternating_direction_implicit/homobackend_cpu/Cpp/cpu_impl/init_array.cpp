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
#include <immintrin.h> // For AVX intrinsics (e.g., AVX2, FMA)

using namespace std;

void init_array(int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  // The original code has an 'if (n >= 1)' guard.
  // If n is less than 1, no initialization should occur.
  if (n < 1) {
    return;
  }

  // Strength reduction: Precompute 1.0/n to replace division with multiplication
  // in the inner loop, which is generally faster.
  const double inv_n = 1.0 / static_cast<double>(n);
  // Load inv_n into an AVX register, broadcasting it to all 4 double lanes.
  const __m256d v_inv_n = _mm256_set1_pd(inv_n);

  // Precompute a constant offset vector for c2 values: [0.0, 1.0, 2.0, 3.0].
  // This vector will be added to a broadcasted 'c2' value to efficiently generate
  // the vector [c2, c2+1, c2+2, c2+3].
  // _mm256_set_pd takes arguments in reverse order (d3, d2, d1, d0) for [d0, d1, d2, d3].
  const __m256d v_offset_c2 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);

  // Outer loop iterates over rows (c1)
  for (int c1 = 0; c1 < n; ++c1) {
    const double dc1 = static_cast<double>(c1);

    // Precompute constant terms for the current row (c1).
    // The original expressions are of the form:
    //   (((double)c1) * (c2 + K) + K) / n
    // This can be rewritten as:
    //   ((dc1 * c2) + (dc1 * K + K)) * inv_n
    // The term (dc1 * K + K) is constant for a given c1 and K.
    // For X (K=1): dc1 * 1 + 1 = dc1 + 1.0
    // For A (K=2): dc1 * 2 + 2 = 2.0 * dc1 + 2.0
    // For B (K=3): dc1 * 3 + 3 = 3.0 * dc1 + 3.0
    const __m256d v_term_X = _mm256_set1_pd(dc1 + 1.0);
    const __m256d v_term_A = _mm256_set1_pd(2.0 * dc1 + 2.0);
    const __m256d v_term_B = _mm256_set1_pd(3.0 * dc1 + 3.0);
    
    // Load dc1 into an AVX register, broadcasting it to all 4 double lanes.
    const __m256d v_dc1 = _mm256_set1_pd(dc1);

    // Inner loop iterates over columns (c2).
    // This loop is vectorized using AVX2 intrinsics, processing 4 'double' elements at a time.
    // A __m256d register holds 4 doubles.
    int c2;
    for (c2 = 0; c2 <= n - 4; c2 += 4) {
      // Create a vector of c2 values: [c2, c2+1, c2+2, c2+3].
      // This is done by broadcasting 'c2' and adding the precomputed offset vector.
      const __m256d v_c2_base = _mm256_set1_pd(static_cast<double>(c2));
      const __m256d v_c2 = _mm256_add_pd(v_c2_base, v_offset_c2);

      // Calculate values for X: (v_dc1 * v_c2 + v_term_X) * v_inv_n
      // _mm256_fmadd_pd performs a Fused Multiply-Add (a*b + c), which is efficient.
      __m256d val_X = _mm256_fmadd_pd(v_dc1, v_c2, v_term_X);
      val_X = _mm256_mul_pd(val_X, v_inv_n);
      // Store the 4 calculated doubles into the X array.
      // Using _mm256_storeu_pd for unaligned store, as the address &X[c1][c2]
      // is not guaranteed to be 32-byte aligned for arbitrary c2.
      _mm256_storeu_pd(&X[c1][c2], val_X);

      // Calculate values for A: (v_dc1 * v_c2 + v_term_A) * v_inv_n
      __m256d val_A = _mm256_fmadd_pd(v_dc1, v_c2, v_term_A);
      val_A = _mm256_mul_pd(val_A, v_inv_n);
      _mm256_storeu_pd(&A[c1][c2], val_A);

      // Calculate values for B: (v_dc1 * v_c2 + v_term_B) * v_inv_n
      __m256d val_B = _mm256_fmadd_pd(v_dc1, v_c2, v_term_B);
      val_B = _mm256_mul_pd(val_B, v_inv_n);
      _mm256_storeu_pd(&B[c1][c2], val_B);
    }

    // Scalar remainder loop for columns.
    // This loop handles any remaining elements if 'n' is not a multiple of 4.
    // A simple loop is used for correctness and clarity, as the performance
    // impact of 0-3 elements is negligible compared to the vectorized part.
    for (; c2 < n; ++c2) {
        const double dc2 = static_cast<double>(c2);
        X[c1][c2] = (dc1 * (dc2 + 1.0) + 1.0) * inv_n;
        A[c1][c2] = (dc1 * (dc2 + 2.0) + 2.0) * inv_n;
        B[c1][c2] = (dc1 * (dc2 + 3.0) + 3.0) * inv_n;
    }
  }
}