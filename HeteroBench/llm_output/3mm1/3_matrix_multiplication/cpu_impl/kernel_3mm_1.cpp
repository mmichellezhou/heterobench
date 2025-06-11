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

using namespace std;

void kernel_3m_1(double C[NJ + 0][NM + 0], double D[NM + 0][NL + 0], double F[NJ + 0][NL + 0])
{
  int c1; // Corresponds to row index for F and C (NJ)
  int c2; // Corresponds to column index for F and D (NL)
  int c5; // Corresponds to inner dimension for C and D (NM)

  // Original loop order: (c1, c2, c5)
  // This order results in non-contiguous memory access for D[c5][c2]
  // (column-major access on a row-major array), leading to poor cache performance.

  // Optimized loop order: (c1, c5, c2)
  // This order ensures contiguous memory access for all arrays, which is crucial for
  // cache efficiency and effective auto-vectorization by the compiler:
  // - C[c1][c5]: c1 (outer), c5 (middle) -> accesses C row-wise (good)
  // - D[c5][c2]: c5 (middle), c2 (inner) -> accesses D row-wise (good)
  // - F[c1][c2]: c1 (outer), c2 (inner) -> accesses F row-wise (good)

  for (c1 = 0; c1 < NJ; c1++) {
    for (c5 = 0; c5 < NM; c5++) {
      // C[c1][c5] is constant within the innermost loop.
      // Loading it once into a register (c_val) reduces redundant memory loads
      // and allows the compiler to optimize the innermost loop more effectively.
      double c_val = C[c1][c5];
      for (c2 = 0; c2 < NL; c2++) {
        F[c1][c2] += c_val * D[c5][c2];
      }
    }
  }
}