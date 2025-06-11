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
#include <immintrin.h> // Required for AVX intrinsics (e.g., _mm256_loadu_pd, _mm256_mul_pd)

using namespace std;

void kernel_3m_1(double C[NJ + 0][NM + 0], double D[NM + 0][NL + 0], double F[NJ + 0][NL + 0])
{
  int c1; // Corresponds to NJ (rows of F, C)
  int c2; // Corresponds to NL (columns of F, D)
  int c5; // Corresponds to NM (columns of C, rows of D)

  // Optimization Strategy:
  // 1. Loop Order Change (Cache Locality):
  //    The original loop order was i-j-k (c1-c2-c5):
  //      for (c1) { for (c2) { for (c5) { F[c1][c2] += C[c1][c5] * D[c5][c2]; } } }
  //    This order results in column-major access for matrix D (D[c5][c2] where c5 is inner, c2 is middle),
  //    which is inefficient for row-major memory layouts (standard C++ arrays).
  //    The optimized loop order is i-k-j (c1-c5-c2):
  //      for (c1) { for (c5) { for (c2) { F[c1][c2] += C[c1][c5] * D[c5][c2]; } } }
  //    This ensures all matrices (F, C, D) are accessed in a row-major fashion,
  //    significantly improving cache utilization and reducing memory latency.
  //
  // 2. Vectorization (SIMD):
  //    The innermost loop (c2) now accesses F[c1][c2] and D[c5][c2] contiguously.
  //    This pattern is ideal for Single Instruction, Multiple Data (SIMD) operations.
  //    AVX intrinsics are used to process 4 double-precision floating-point numbers
  //    simultaneously, leveraging modern CPU capabilities.
  //
  // 3. Scalar Broadcast:
  //    The value C[c1][c5] is constant within the innermost c2 loop. It is loaded
  //    once into a scalar variable (`c_val`) and then broadcast into a SIMD register
  //    (`c_vec`) to be multiplied with vector elements of D, avoiding redundant loads.

  // Define vectorization width for double precision (AVX: 4 doubles per __m256d register)
  const int VEC_SIZE = 4; 

  for (c1 = 0; c1 < NJ; c1++) { // Outer loop: iterates over rows of F and C
    for (c5 = 0; c5 < NM; c5++) { // Middle loop: iterates over columns of C and rows of D
      // Load C[c1][c5] once into a scalar variable.
      double c_val = C[c1][c5];
      // Broadcast the scalar c_val to all elements of a vector register.
      __m256d c_vec = _mm256_set1_pd(c_val); 

      // Innermost loop: iterates over columns of F and D. This loop is vectorized.
      // The loop iterates in steps of VEC_SIZE for the main vectorized part.
      int c2;
      for (c2 = 0; c2 < NL - (NL % VEC_SIZE); c2 += VEC_SIZE) {
        // Load 4 doubles from F. _mm256_loadu_pd is used for unaligned loads,
        // which is safer for general C++ array allocations.
        __m256d f_vec = _mm256_loadu_pd(&F[c1][c2]); 
        // Load 4 doubles from D.
        __m256d d_vec = _mm256_loadu_pd(&D[c5][c2]);

        // Perform vectorized multiplication: c_vec * d_vec
        __m256d prod_vec = _mm256_mul_pd(c_vec, d_vec);
        // Perform vectorized addition: f_vec + prod_vec
        f_vec = _mm256_add_pd(f_vec, prod_vec);

        // Store the result back to F.
        _mm256_storeu_pd(&F[c1][c2], f_vec);
      }

      // Handle remaining elements if NL is not a multiple of VEC_SIZE.
      // This ensures correctness for all matrix dimensions.
      for (; c2 < NL; c2++) {
        F[c1][c2] += c_val * D[c5][c2];
      }
    }
  }
}