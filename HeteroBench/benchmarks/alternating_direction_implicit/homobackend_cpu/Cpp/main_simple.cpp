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

void golden_init_array(int n, double X[N + 0][N + 0], double A[N + 0][N + 0],
                       double B[N + 0][N + 0]) {
  {
    int c1;
    int c2;
    if (n >= 1) {
      for (c1 = 0; c1 <= n + -1; c1++) {
        for (c2 = 0; c2 <= n + -1; c2++) {
          X[c1][c2] = (((double)c1) * (c2 + 1) + 1) / n;
          A[c1][c2] = (((double)c1) * (c2 + 2) + 2) / n;
          B[c1][c2] = (((double)c1) * (c2 + 3) + 3) / n;
        }
      }
    }
  }
}

void golden_kernel_adi(int tsteps, int n, double X[N + 0][N + 0],
                       double A[N + 0][N + 0], double B[N + 0][N + 0]) {
  {
    int c0;
    int c2;
    int c8;
    for (c0 = 0; c0 <= TSTEPS; c0++) {
      for (c2 = 0; c2 <= N - 1; c2++) {
        for (c8 = 1; c8 <= N - 1; c8++) {
          B[c2][c8] = B[c2][c8] - A[c2][c8] * A[c2][c8] / B[c2][c8 - 1];
        }
        for (c8 = 1; c8 <= N - 1; c8++) {
          X[c2][c8] = X[c2][c8] - X[c2][c8 - 1] * A[c2][c8] / B[c2][c8 - 1];
        }
        for (c8 = 0; c8 <= N - 3; c8++) {
          X[c2][N - c8 - 2] =
              (X[c2][N - 2 - c8] - X[c2][N - 2 - c8 - 1] * A[c2][N - c8 - 3]) /
              B[c2][N - 3 - c8];
        }
      }
      for (c2 = 0; c2 <= N - 1; c2++) {
        X[c2][N - 1] = X[c2][N - 1] / B[c2][N - 1];
      }
      for (c2 = 0; c2 <= N - 1; c2++) {
        for (c8 = 1; c8 <= N - 1; c8++) {
          B[c8][c2] = B[c8][c2] - A[c8][c2] * A[c8][c2] / B[c8 - 1][c2];
        }
        for (c8 = 1; c8 <= N - 1; c8++) {
          X[c8][c2] = X[c8][c2] - X[c8 - 1][c2] * A[c8][c2] / B[c8 - 1][c2];
        }
        for (c8 = 0; c8 <= N - 3; c8++) {
          X[N - 2 - c8][c2] =
              (X[N - 2 - c8][c2] - X[N - c8 - 3][c2] * A[N - 3 - c8][c2]) /
              B[N - 2 - c8][c2];
        }
      }
      for (c2 = 0; c2 <= N - 1; c2++) {
        X[N - 1][c2] = X[N - 1][c2] / B[N - 1][c2];
      }
    }
  }
}

int main(int argc, char **argv) {
  cout << "=======================================" << endl;
  cout << "Running adi benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  double(*X)[N + 0][N + 0];
  X = ((double(*)[N + 0][N + 0])(
      polybench_alloc_data(((N + 0) * (N + 0)), (sizeof(double)))));
  double(*A)[N + 0][N + 0];
  A = ((double(*)[N + 0][N + 0])(
      polybench_alloc_data(((N + 0) * (N + 0)), (sizeof(double)))));
  double(*B)[N + 0][N + 0];
  B = ((double(*)[N + 0][N + 0])(
      polybench_alloc_data(((N + 0) * (N + 0)), (sizeof(double)))));

  // Allocate arrays for golden version
  double(*X_golden)[N + 0][N + 0];
  X_golden = ((double(*)[N + 0][N + 0])(
      polybench_alloc_data(((N + 0) * (N + 0)), (sizeof(double)))));
  double(*A_golden)[N + 0][N + 0];
  A_golden = ((double(*)[N + 0][N + 0])(
      polybench_alloc_data(((N + 0) * (N + 0)), (sizeof(double)))));
  double(*B_golden)[N + 0][N + 0];
  B_golden = ((double(*)[N + 0][N + 0])(
      polybench_alloc_data(((N + 0) * (N + 0)), (sizeof(double)))));

  /* Correctness tests. */
  // Run golden implementation for correctness check
  golden_init_array(n, *X_golden, *A_golden, *B_golden);
  golden_kernel_adi(tsteps, n, *X_golden, *A_golden, *B_golden);

  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  init_array(n, *X, *A, *B);
  kernel_adi(tsteps, n, *X, *A, *B);
  cout << "Done" << endl;

  // Check original implementation results
  cout << "Checking original implementation results..." << endl;
  int error = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (abs((*X)[i][j] - (*X_golden)[i][j]) > 0.1) {
        error += 1;
      }
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (abs((*B)[i][j] - (*B_golden)[i][j]) > 0.1) {
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
  init_array_optimized(n, *X, *A, *B);
  kernel_adi_optimized(tsteps, n, *X, *A, *B);
  cout << "Done" << endl;

  // Check optimized implementation results
  cout << "Checking optimized implementation results..." << endl;
  error = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (abs((*X)[i][j] - (*X_golden)[i][j]) > 0.1) {
        error += 1;
      }
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (abs((*B)[i][j] - (*B_golden)[i][j]) > 0.1) {
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
  double init_array_time = 0;
  double kernel_adi_time = 0;
  double init_array_optimized_time = 0;
  double kernel_adi_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    init_array(n, *X, *A, *B);
    init_array_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    kernel_adi(tsteps, n, *X, *A, *B);
    kernel_adi_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    init_array_optimized(n, *X, *A, *B);
    init_array_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    kernel_adi_optimized(tsteps, n, *X, *A, *B);
    kernel_adi_optimized_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;

  double original_total_time = init_array_time + kernel_adi_time;
  double optimized_total_time =
      init_array_optimized_time + kernel_adi_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  init_array time: " << init_array_time / iterations << " seconds"
       << endl;
  cout << "  kernel_adi time: " << kernel_adi_time / iterations << " seconds"
       << endl;
  cout << "  Single iteration time: " << original_total_time / iterations
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  init_array time: " << init_array_optimized_time / iterations
       << " seconds" << endl;
  cout << "  kernel_adi time: " << kernel_adi_optimized_time / iterations
       << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / iterations
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  init_array: " << init_array_time / init_array_optimized_time << "x"
       << endl;
  cout << "  kernel_adi: " << kernel_adi_time / kernel_adi_optimized_time << "x"
       << endl;
  cout << "  Total: " << original_total_time / optimized_total_time << "x"
       << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  free(((void *)A));
  free(((void *)B));
  free(((void *)X));
  free(((void *)A_golden));
  free(((void *)B_golden));
  free(((void *)X_golden));

  return 0;
}