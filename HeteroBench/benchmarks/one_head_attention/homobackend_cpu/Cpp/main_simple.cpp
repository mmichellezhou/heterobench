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

bool checkFileExistence(const std::string &filePath) {
  std::ifstream file(filePath);
  if (!file) {
    cout << "Error: File " << filePath << " does not exist or cannot be opened."
         << endl;
    cout << "You may need to run the Python version of this benchmark to "
            "generate the data file first."
         << endl;
    return false;
  }
  return true;
}

/* Read data from file*/
void readData(const char *file_path_mat, double *data_mat, size_t size_mat) {
  // cout << "Reading " << file_path_mat << " ... ";
  if (!checkFileExistence(file_path_mat)) {
    exit(1);
  }
  std::ifstream in_mat(file_path_mat, std::ios::in | std::ios::binary);
  in_mat.read((char *)data_mat, sizeof(double) * size_mat);
  in_mat.close();
  // cout << "Done" << endl;
}

/* Check results */
void checkResult(double *data, double *data_golden, size_t size) {
  int error = 0;
  for (int i = 0; i < size; i++) {
    if (abs(data[i] - data_golden[i]) > 1e-2) {
      error++;
    }
  }
  if (error == 0) {
    cout << "Correct" << endl;
  } else {
    cout << "Wrong" << endl;
    cout << "error: " << error << endl;

    // print the first 10 elements of computed results
    cout << "First 10 elements of computed results: ";
    for (int i = 0; i < 10; i++) {
      cout << data[i] << " ";
    }
    cout << endl;

    // print the first 10 elements of golden results
    cout << "First 10 elements of golden results: ";
    for (int i = 0; i < 10; i++) {
      cout << data_golden[i] << " ";
    }
    cout << endl;
  }
}

/* Compare original and optimized results */
void compareResults(double *original, double *optimized, size_t size,
                    const string &name) {
  int error = 0;
  for (int i = 0; i < size; i++) {
    if (abs(original[i] - optimized[i]) > 1e-10) {
      error++;
    }
  }
  if (!error) {
    cout << name << ": Pass (Original and optimized match)" << endl;
  } else {
    cout << name << ": Fail (Original and optimized differ)" << endl;
    cout << "error: " << error << " differences found" << endl;

    // print the first 10 elements of original results
    cout << "First 10 elements of original results: ";
    for (int i = 0; i < 10; i++) {
      cout << original[i] << " ";
    }
    cout << endl;

    // print the first 10 elements of optimized results
    cout << "First 10 elements of optimized results: ";
    for (int i = 0; i < 10; i++) {
      cout << optimized[i] << " ";
    }
    cout << endl;
  }
}

int main(int argc, char *argv[]) {
  cout << "=======================================" << endl;
  cout << "Running one_head_attention benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  string input_path;
  string output_path;

  if (argc == 3) {
    input_path = argv[1];
    output_path = argv[2];
  } else {
    printf("Usage: ./oha_sw <input_path> <output_path>\n");
    exit(-1);
  }

  /* Variable declaration/allocation. */
  double *query = new double[BATCH_SIZE * N * D];
  double *key = new double[BATCH_SIZE * N * D];
  double *value = new double[BATCH_SIZE * N * D];
  double *transpose_key = new double[BATCH_SIZE * D * N];
  double *matmul_query_key = new double[BATCH_SIZE * N * N];
  double *softmax_result = new double[BATCH_SIZE * N * N];
  double *matmul_softmax_value = new double[BATCH_SIZE * N * D];

  // Allocate array for optimized implementation
  double *transpose_key_optimized = new double[BATCH_SIZE * D * N];
  double *matmul_query_key_optimized = new double[BATCH_SIZE * N * N];
  double *softmax_result_optimized = new double[BATCH_SIZE * N * N];
  double *matmul_softmax_value_optimized = new double[BATCH_SIZE * N * D];

  /*
  // Allocate array for golden implementation
  double *transpose_key_golden = new double[BATCH_SIZE * D * N];
  double *matmul_query_key_golden = new double[BATCH_SIZE * N * N];
  double *softmax_result_golden = new double[BATCH_SIZE * N * N];
  double *matmul_softmax_value_golden = new double[BATCH_SIZE * N * D];
  */

  /* Correctness tests. */
  /*
  // Run golden implementation from python version
  readData((output_path + "/transpose_key_golden.bin").c_str(),
           transpose_key_golden, BATCH_SIZE * D * N);
  readData((output_path + "/matmul_query_key_golden.bin").c_str(),
           matmul_query_key_golden, BATCH_SIZE * N * N);
  readData((output_path + "/softmax_result_golden.bin").c_str(),
           softmax_result_golden, BATCH_SIZE * N * N);
  readData((output_path + "/matmul_softmax_value_golden.bin").c_str(),
           matmul_softmax_value_golden, BATCH_SIZE * N * D);
  */

  // // init matmul_query_key to 0
  // for (int i = 0; i < BATCH_SIZE * N * N; i++) {
  //   matmul_query_key[i] = 0;
  // }

  // // init matmul_softmax_value to 0
  // for (int i = 0; i < BATCH_SIZE * N * D; i++) {
  //   matmul_softmax_value[i] = 0;
  // }

  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  readData((input_path + "/query.bin").c_str(), query, BATCH_SIZE * N * D);
  readData((input_path + "/key.bin").c_str(), key, BATCH_SIZE * N * D);
  readData((input_path + "/value.bin").c_str(), value, BATCH_SIZE * N * D);
  transpose(key, transpose_key, BATCH_SIZE, N, D, -2, -1);
  matmul(query, transpose_key, matmul_query_key, BATCH_SIZE, N, D, N);
  softmax(matmul_query_key, softmax_result, BATCH_SIZE, N, N, -1);
  matmul(softmax_result, value, matmul_softmax_value, BATCH_SIZE, N, N, D);
  cout << "Done" << endl;

  /*
  // Check original implementation results
  cout << "Checking original implementation results..." << endl;
  cout << "Checking transpose_key... ";
  checkResult(transpose_key, transpose_key_golden, BATCH_SIZE * D * N);
  cout << "Checking matmul_query_key... ";
  checkResult(matmul_query_key, matmul_query_key_golden, BATCH_SIZE * N * N);
  cout << "Checking softmax_result... ";
  checkResult(softmax_result, softmax_result_golden, BATCH_SIZE * N * N);
  cout << "Checking matmul_softmax_value... ";
  checkResult(matmul_softmax_value, matmul_softmax_value_golden,
              BATCH_SIZE * N * D);
  */

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  transpose_optimized(key, transpose_key_optimized, BATCH_SIZE, N, D, -2, -1);
  matmul_optimized(query, transpose_key_optimized, matmul_query_key_optimized,
                   BATCH_SIZE, N, D, N);
  softmax_optimized(matmul_query_key_optimized, softmax_result_optimized,
                    BATCH_SIZE, N, N, -1);
  matmul_optimized(softmax_result_optimized, value,
                   matmul_softmax_value_optimized, BATCH_SIZE, N, N, D);
  cout << "Done" << endl;

  /*
  // Check optimized implementation results
  cout << "Checking optimized implementation results..." << endl;
  cout << "Checking transpose_key... ";
  checkResult(transpose_key, transpose_key_golden, BATCH_SIZE * D * N);
  cout << "Checking matmul_query_key... ";
  checkResult(matmul_query_key, matmul_query_key_golden, BATCH_SIZE * N * N);
  cout << "Checking softmax_result... ";
  checkResult(softmax_result, softmax_result_golden, BATCH_SIZE * N * N);
  cout << "Checking matmul_softmax_value... ";
  checkResult(matmul_softmax_value, matmul_softmax_value_golden,
              BATCH_SIZE * N * D);
  */

  // Compare original and optimized results
  cout << "Comparing original and optimized results..." << endl;
  compareResults(transpose_key, transpose_key_optimized, BATCH_SIZE * D * N,
                 "transpose_key");
  compareResults(matmul_query_key, matmul_query_key_optimized,
                 BATCH_SIZE * N * N, "matmul_query_key");
  compareResults(softmax_result, softmax_result_optimized, BATCH_SIZE * N * N,
                 "softmax_result");
  compareResults(matmul_softmax_value, matmul_softmax_value_optimized,
                 BATCH_SIZE * N * D, "matmul_softmax_value");

  /* Performance measurement. */
  int iterations = ITERATIONS;
  cout << "Running " << iterations
       << " iterations for performance measurement..." << endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double transpose_time = 0;
  double matmul_query_key_time = 0;
  double softmax_time = 0;
  double matmul_softmax_value_time = 0;
  double transpose_optimized_time = 0;
  double matmul_query_key_optimized_time = 0;
  double softmax_optimized_time = 0;
  double matmul_softmax_value_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    transpose(key, transpose_key, BATCH_SIZE, N, D, -2, -1);
    transpose_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    matmul(query, transpose_key, matmul_query_key, BATCH_SIZE, N, D, N);
    matmul_query_key_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    softmax(matmul_query_key, softmax_result, BATCH_SIZE, N, N, -1);
    softmax_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    matmul(softmax_result, value, matmul_softmax_value, BATCH_SIZE, N, N, D);
    matmul_softmax_value_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    transpose_optimized(key, transpose_key_optimized, BATCH_SIZE, N, D, -2, -1);
    transpose_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    matmul_optimized(query, transpose_key_optimized, matmul_query_key_optimized,
                     BATCH_SIZE, N, D, N);
    matmul_query_key_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    softmax_optimized(matmul_query_key_optimized, softmax_result_optimized,
                      BATCH_SIZE, N, N, -1);
    softmax_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    matmul_optimized(softmax_result_optimized, value,
                     matmul_softmax_value_optimized, BATCH_SIZE, N, N, D);
    matmul_softmax_value_optimized_time +=
        omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;

  double original_total_time = transpose_time + matmul_query_key_time +
                               softmax_time + matmul_softmax_value_time;
  double optimized_total_time =
      transpose_optimized_time + matmul_query_key_optimized_time +
      softmax_optimized_time + matmul_softmax_value_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  transpose time: " << transpose_time / iterations << " seconds"
       << endl;
  cout << "  matmul_1 time: " << matmul_query_key_time / iterations
       << " seconds" << endl;
  cout << "  softmax time: " << softmax_time / iterations << " seconds" << endl;
  cout << "  matmul_2 time: " << matmul_softmax_value_time / iterations
       << " seconds" << endl;
  cout << "  Single iteration time: " << original_total_time / iterations
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  transpose time: " << transpose_optimized_time / iterations
       << " seconds" << endl;
  cout << "  matmul_1 time: " << matmul_query_key_optimized_time / iterations
       << " seconds" << endl;
  cout << "  softmax time: " << softmax_optimized_time / iterations
       << " seconds" << endl;
  cout << "  matmul_2 time: "
       << matmul_softmax_value_optimized_time / iterations << " seconds"
       << endl;
  cout << "  Single iteration time: " << optimized_total_time / iterations
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  transpose: " << transpose_time / transpose_optimized_time << endl;
  cout << "  matmul_1: "
       << matmul_query_key_time / matmul_query_key_optimized_time << endl;
  cout << "  softmax: " << softmax_time / softmax_optimized_time << endl;
  cout << "  matmul_2: "
       << matmul_softmax_value_time / matmul_softmax_value_optimized_time
       << endl;
  cout << "  Total: " << original_total_time / optimized_total_time << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  delete[] query;
  delete[] key;
  delete[] value;
  delete[] transpose_key;
  delete[] matmul_query_key;
  delete[] softmax_result;
  delete[] matmul_softmax_value;
  delete[] transpose_key_optimized;
  delete[] matmul_query_key_optimized;
  delete[] softmax_result_optimized;
  delete[] matmul_softmax_value_optimized;
  /*
  delete[] transpose_key_golden;
  delete[] matmul_query_key_golden;
  delete[] softmax_result_golden;
  delete[] matmul_softmax_value_golden;
  */

  return 0;
}