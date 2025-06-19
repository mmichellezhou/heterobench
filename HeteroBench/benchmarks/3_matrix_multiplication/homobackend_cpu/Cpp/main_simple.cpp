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
#include "init_array.h"
#include "omp.h"

#include "cpu_impl_optimized.h"

// standard C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace std;

void *xmalloc(size_t num) {
  void *nnew = NULL;
  int ret = posix_memalign(&nnew, 32, num);
  if (!nnew || ret) {
    fprintf(stderr, "[PolyBench] posix_memalign: cannot allocate memory");
    exit(1);
  }
  return nnew;
}

void *polybench_alloc_data(unsigned long long int n, int elt_size) {
  /// FIXME: detect overflow!
  size_t val = n;
  val *= elt_size;
  void *ret = xmalloc(val);

  return ret;
}

void golden_kernel_3mm(double E[NI + 0][NJ + 0], double A[NI + 0][NK + 0],
                       double B[NK + 0][NJ + 0], double F[NJ + 0][NL + 0],
                       double C[NJ + 0][NM + 0], double D[NM + 0][NL + 0],
                       double G[NI + 0][NL + 0]) {
  int c1;
  int c2;
  int c5;
  int c6;
  // #pragma omp parallel for private(c2)
  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      G[c1][c2] = 0;
    }
  }
  // #pragma omp parallel for private(c2)
  for (c1 = 0; c1 <= NJ - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      F[c1][c2] = 0;
    }
  }
  // #pragma omp parallel for private(c5, c2)
  for (c1 = 0; c1 <= NJ - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      for (c5 = 0; c5 <= NM - 1; c5++) {
        F[c1][c2] += C[c1][c5] * D[c5][c2];
      }
    }
  }
  // #pragma omp parallel for private(c2)
  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      E[c1][c2] = 0;
    }
  }
  // #pragma omp parallel for private(c5, c2)
  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      for (c5 = 0; c5 <= NK - 1; c5++) {
        E[c1][c2] += A[c1][c5] * B[c5][c2];
      }
      for (c6 = 0; c6 <= NL - 1; c6++) {
        G[c1][c6] += E[c1][c2] * F[c2][c6];
      }
    }
  }

  //// #pragma endscop
}

void reinit_output_array(double E[NI + 0][NJ + 0], double F[NJ + 0][NL + 0],
                         double G[NI + 0][NL + 0]) {
  int c1, c2;

  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      G[c1][c2] = 0;
    }
  }

  for (c1 = 0; c1 <= NJ - 1; c1++) {
    for (c2 = 0; c2 <= NL - 1; c2++) {
      F[c1][c2] = 0;
    }
  }

  for (c1 = 0; c1 <= NI - 1; c1++) {
    for (c2 = 0; c2 <= NJ - 1; c2++) {
      E[c1][c2] = 0;
    }
  }
}

int main(int argc, char **argv) {
  cout << "=======================================" << endl;
  cout << "Running 3mm benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  int nm = NM;

  /* Variable declaration/allocation. */
  double(*E)[NI + 0][NJ + 0];
  E = ((double(*)[NI + 0][NJ + 0])(
      polybench_alloc_data(((NI + 0) * (NJ + 0)), (sizeof(double)))));
  double(*A)[NI + 0][NK + 0];
  A = ((double(*)[NI + 0][NK + 0])(
      polybench_alloc_data(((NI + 0) * (NK + 0)), (sizeof(double)))));
  double(*B)[NK + 0][NJ + 0];
  B = ((double(*)[NK + 0][NJ + 0])(
      polybench_alloc_data(((NK + 0) * (NJ + 0)), (sizeof(double)))));
  double(*F)[NJ + 0][NL + 0];
  F = ((double(*)[NJ + 0][NL + 0])(
      polybench_alloc_data(((NJ + 0) * (NL + 0)), (sizeof(double)))));
  double(*C)[NJ + 0][NM + 0];
  C = ((double(*)[NJ + 0][NM + 0])(
      polybench_alloc_data(((NJ + 0) * (NM + 0)), (sizeof(double)))));
  double(*D)[NM + 0][NL + 0];
  D = ((double(*)[NM + 0][NL + 0])(
      polybench_alloc_data(((NM + 0) * (NL + 0)), (sizeof(double)))));
  double(*G)[NI + 0][NL + 0];
  G = ((double(*)[NI + 0][NL + 0])(
      polybench_alloc_data(((NI + 0) * (NL + 0)), (sizeof(double)))));

  // Allocate array for golden implementation
  double(*G_golden)[NI + 0][NL + 0];
  G_golden = ((double(*)[NI + 0][NL + 0])(
      polybench_alloc_data(((NI + 0) * (NL + 0)), (sizeof(double)))));

  /* Correctness tests. */
  // Run golden implementation
  init_array(ni, nj, nk, nl, nm, *A, *B, *C, *D, *E, *F, *G);
  golden_kernel_3mm(*E, *A, *B, *F, *C, *D, *G_golden);

  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  reinit_output_array(*E, *F, *G);
  kernel_3m_0(*A, *B, *E);
  kernel_3m_1(*C, *D, *F);
  kernel_3m_2(*E, *F, *G);
  cout << "Done" << endl;

  // Check original implementation results
  cout << "Checking original implementation results..." << endl;
  int error = 0;
  for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NL; j++) {
      if (abs((*G)[i][j] - (*G_golden)[i][j]) > 0.1) {
        error += 1;
      }
    }
  }
  if (!error) {
    cout << "Original implementation output is correct!" << endl;
  } else {
    cout << "Original implementation output is incorrect!" << endl;
    cout << "Total " << error << " errors!" << endl;
  }

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  reinit_output_array(*E, *F, *G);
  kernel_3m_0_optimized(*A, *B, *E);
  kernel_3m_1_optimized(*C, *D, *F);
  kernel_3m_2_optimized(*E, *F, *G);
  cout << "Done" << endl;

  // Check optimized implementation results
  cout << "Checking optimized implementation results..." << endl;
  error = 0;
  for (int i = 0; i < NI; i++) {
    for (int j = 0; j < NL; j++) {
      if (abs((*G)[i][j] - (*G_golden)[i][j]) > 0.1) {
        error += 1;
      }
    }
  }
  if (!error) {
    cout << "Optimized implementation output is correct!" << endl;
  } else {
    cout << "Optimized implementation output is incorrect!" << endl;
    cout << "Total " << error << " errors!" << endl;
  }

  /* Performance measurement. */
  int iterations = ITERATIONS;
  cout << "Running " << iterations
       << " iterations for performance measurement..." << endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double kernel_3m_0_time = 0;
  double kernel_3m_1_time = 0;
  double kernel_3m_2_time = 0;
  double kernel_3m_0_optimized_time = 0;
  double kernel_3m_1_optimized_time = 0;
  double kernel_3m_2_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int i = 0; i < iterations; i++) {

    start_iteration_time = omp_get_wtime();
    kernel_3m_0(*A, *B, *E);
    kernel_3m_0_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    kernel_3m_1(*C, *D, *F);
    kernel_3m_1_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    kernel_3m_2(*E, *F, *G);
    kernel_3m_2_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    kernel_3m_0_optimized(*A, *B, *E);
    kernel_3m_0_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    kernel_3m_1_optimized(*C, *D, *F);
    kernel_3m_1_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    kernel_3m_2_optimized(*E, *F, *G);
    kernel_3m_2_optimized_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;

  double original_total_time =
      kernel_3m_0_time + kernel_3m_1_time + kernel_3m_2_time;
  double optimized_total_time = kernel_3m_0_optimized_time +
                                kernel_3m_1_optimized_time +
                                kernel_3m_2_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  kernel_3m_0 time: " << (kernel_3m_0_time / iterations)
       << " seconds" << endl;
  cout << "  kernel_3m_1 time: " << (kernel_3m_1_time / iterations)
       << " seconds" << endl;
  cout << "  kernel_3m_2 time: " << (kernel_3m_2_time / iterations)
       << " seconds" << endl;
  cout << "  Single iteration time: " << original_total_time / iterations
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  kernel_3m_0 time: " << (kernel_3m_0_optimized_time / iterations)
       << " seconds" << endl;
  cout << "  kernel_3m_1 time: " << (kernel_3m_1_optimized_time / iterations)
       << " seconds" << endl;
  cout << "  kernel_3m_2 time: " << (kernel_3m_2_optimized_time / iterations)
       << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / iterations
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  kernel_3m_0: " << (kernel_3m_0_time / kernel_3m_0_optimized_time)
       << "x" << endl;
  cout << "  kernel_3m_1: " << (kernel_3m_1_time / kernel_3m_1_optimized_time)
       << "x" << endl;
  cout << "  kernel_3m_2: " << (kernel_3m_2_time / kernel_3m_2_optimized_time)
       << "x" << endl;
  cout << "  Total: " << (original_total_time / optimized_total_time) << "x"
       << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  free(((void *)E));
  free(((void *)A));
  free(((void *)B));
  free(((void *)F));
  free(((void *)C));
  free(((void *)D));
  free(((void *)G));
  free(((void *)G_golden));

  return 0;
}