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

bool checkFileExistence(const string &filePath) {
  ifstream file(filePath);
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
  ifstream in_mat(file_path_mat, ios::in | ios::binary);
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
void compareResults(double *original, double *optimized, int size,
                    const string &name) {
  int error = 0;
  for (int i = 0; i < size; i++) {
    if (abs(original[i] - optimized[i]) > 1e-6) {
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
    for (int i = 0; i < 10 && i < size; i++) {
      cout << original[i] << " ";
    }
    cout << endl;

    // print the first 10 elements of optimized results
    cout << "First 10 elements of optimized results: ";
    for (int i = 0; i < 10 && i < size; i++) {
      cout << optimized[i] << " ";
    }
    cout << endl;
  }
}

void mlp_forward(string input_path, string output_path) {
  double *data_a0 = new double[SIZE_A0];
  double *data_a1 = new double[SIZE_A1];
  double *data_z1 = new double[SIZE_Z1];
  double *data_a2 = new double[SIZE_A2];
  double *data_z2 = new double[SIZE_Z2];
  double *data_a3 = new double[SIZE_A3];
  double *data_z3 = new double[SIZE_Z3];
  double *data_a4 = new double[SIZE_A4];
  double *data_a4_exp = new double[SIZE_A4];
  double *data_z4 = new double[SIZE_Z4];

  // Arrays for optimized implementation results
  double *data_a1_opt = new double[SIZE_A1];
  double *data_z1_opt = new double[SIZE_Z1];
  double *data_a2_opt = new double[SIZE_A2];
  double *data_z2_opt = new double[SIZE_Z2];
  double *data_a3_opt = new double[SIZE_A3];
  double *data_z3_opt = new double[SIZE_Z3];
  double *data_a4_opt = new double[SIZE_A4];
  double *data_z4_opt = new double[SIZE_Z4];

  double *data_w0 = new double[SIZE_W0];
  double *data_w1 = new double[SIZE_W1];
  double *data_w2 = new double[SIZE_W2];
  double *data_w3 = new double[SIZE_W3];

  double *data_b0 = new double[SIZE_B0];
  double *data_b1 = new double[SIZE_B1];
  double *data_b2 = new double[SIZE_B2];
  double *data_b3 = new double[SIZE_B3];

  // readData("../../dataset/a0.bin", data_a0, SIZE_A0);
  // readData("../../dataset/w0.bin", data_w0, SIZE_W0);
  // readData("../../dataset/b0.bin", data_b0, SIZE_B0);
  // readData("../../dataset/w1.bin", data_w1, SIZE_W1);
  // readData("../../dataset/b1.bin", data_b1, SIZE_B1);
  // readData("../../dataset/w2.bin", data_w2, SIZE_W2);
  // readData("../../dataset/b2.bin", data_b2, SIZE_B2);
  // readData("../../dataset/w3.bin", data_w3, SIZE_W3);
  // readData("../../dataset/b3.bin", data_b3, SIZE_B3);
  readData((input_path + "/a0.bin").c_str(), data_a0, SIZE_A0);
  readData((input_path + "/w0.bin").c_str(), data_w0, SIZE_W0);
  readData((input_path + "/b0.bin").c_str(), data_b0, SIZE_B0);
  readData((input_path + "/w1.bin").c_str(), data_w1, SIZE_W1);
  readData((input_path + "/b1.bin").c_str(), data_b1, SIZE_B1);
  readData((input_path + "/w2.bin").c_str(), data_w2, SIZE_W2);
  readData((input_path + "/b2.bin").c_str(), data_b2, SIZE_B2);
  readData((input_path + "/w3.bin").c_str(), data_w3, SIZE_W3);
  readData((input_path + "/b3.bin").c_str(), data_b3, SIZE_B3);

  /* Correctness tests. */
  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  dot_add(data_a0, data_w0, data_b0, data_a1, L0_H1, L0_W1, L0_W1, L0_W2);
  for (int i = 0; i < SIZE_A1; i++) {
    data_a1[i] = data_a1[i] / 500;
  }
  sigmoid(data_a1, data_z1, SIZE_Z1);
  dot_add(data_z1, data_w1, data_b1, data_a2, L1_H1, L1_W1, L1_W1, L1_W2);
  for (int i = 0; i < SIZE_A2; i++) {
    data_a2[i] = data_a2[i] / 1500;
  }
  sigmoid(data_a2, data_z2, SIZE_Z2);
  dot_add(data_z2, data_w2, data_b2, data_a3, L2_H1, L2_W1, L2_W1, L2_W2);
  for (int i = 0; i < SIZE_A3; i++) {
    data_a3[i] = data_a3[i] / 1500;
  }
  sigmoid(data_a3, data_z3, SIZE_Z3);
  dot_add(data_z3, data_w3, data_b3, data_a4, L3_H1, L3_W1, L3_W1, L3_W2);
  for (int i = 0; i < SIZE_A4; i++) {
    data_a4[i] = data_a4[i] / 1500;
  }
  softmax(data_a4, data_a4_exp, data_z4, SIZE_Z4);
  for (int i = 0; i < SIZE_A4; i++) {
    data_z4[i] = data_z4[i] * 1000000;
  }
  cout << "Done" << endl;

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  dot_add_optimized(data_a0, data_w0, data_b0, data_a1_opt, L0_H1, L0_W1, L0_W1,
                    L0_W2);
  for (int i = 0; i < SIZE_A1; i++) {
    data_a1_opt[i] = data_a1_opt[i] / 500;
  }
  sigmoid_optimized(data_a1_opt, data_z1_opt, SIZE_Z1);
  dot_add_optimized(data_z1_opt, data_w1, data_b1, data_a2_opt, L1_H1, L1_W1,
                    L1_W1, L1_W2);
  for (int i = 0; i < SIZE_A2; i++) {
    data_a2_opt[i] = data_a2_opt[i] / 1500;
  }
  sigmoid_optimized(data_a2_opt, data_z2_opt, SIZE_Z2);
  dot_add_optimized(data_z2_opt, data_w2, data_b2, data_a3_opt, L2_H1, L2_W1,
                    L2_W1, L2_W2);
  for (int i = 0; i < SIZE_A3; i++) {
    data_a3_opt[i] = data_a3_opt[i] / 1500;
  }
  sigmoid_optimized(data_a3_opt, data_z3_opt, SIZE_Z3);
  dot_add_optimized(data_z3_opt, data_w3, data_b3, data_a4_opt, L3_H1, L3_W1,
                    L3_W1, L3_W2);
  for (int i = 0; i < SIZE_A4; i++) {
    data_a4_opt[i] = data_a4_opt[i] / 1500;
  }
  softmax_optimized(data_a4_opt, data_a4_exp, data_z4_opt, SIZE_Z4);
  for (int i = 0; i < SIZE_A4; i++) {
    data_z4_opt[i] = data_z4_opt[i] * 1000000;
  }
  cout << "Done" << endl;

  // Compare results
  cout << "Comparing original and optimized results..." << endl;
  compareResults(data_a1, data_a1_opt, SIZE_A1, "Layer 0 (dot_add)");
  compareResults(data_z1, data_z1_opt, SIZE_Z1, "Layer 0 (sigmoid)");
  compareResults(data_a2, data_a2_opt, SIZE_A2, "Layer 1 (dot_add)");
  compareResults(data_z2, data_z2_opt, SIZE_Z2, "Layer 1 (sigmoid)");
  compareResults(data_a3, data_a3_opt, SIZE_A3, "Layer 2 (dot_add)");
  compareResults(data_z3, data_z3_opt, SIZE_Z3, "Layer 2 (sigmoid)");
  compareResults(data_a4, data_a4_opt, SIZE_A4, "Layer 3 (dot_add)");
  compareResults(data_z4, data_z4_opt, SIZE_Z4, "Layer 3 (softmax)");

  /* Performance measurement. */
  int iterations = ITERATIONS;
  cout << "Running " << iterations
       << " iterations for performance measurement..." << endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double layer_0_time = 0;
  double layer_1_time = 0;
  double layer_2_time = 0;
  double layer_3_time = 0;
  double layer_0_optimized_time = 0;
  double layer_1_optimized_time = 0;
  double layer_2_optimized_time = 0;
  double layer_3_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    dot_add(data_a0, data_w0, data_b0, data_a1, L0_H1, L0_W1, L0_W1, L0_W2);
    for (int j = 0; j < SIZE_A1; j++) {
      data_a1[j] = data_a1[j] / 500;
    }
    sigmoid(data_a1, data_z1, SIZE_Z1);
    layer_0_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add(data_z1, data_w1, data_b1, data_a2, L1_H1, L1_W1, L1_W1, L1_W2);
    for (int j = 0; j < SIZE_A2; j++) {
      data_a2[j] = data_a2[j] / 1500;
    }
    sigmoid(data_a2, data_z2, SIZE_Z2);
    layer_1_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add(data_z2, data_w2, data_b2, data_a3, L2_H1, L2_W1, L2_W1, L2_W2);
    for (int j = 0; j < SIZE_A3; j++) {
      data_a3[j] = data_a3[j] / 1500;
    }
    sigmoid(data_a3, data_z3, SIZE_Z3);
    layer_2_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add(data_z3, data_w3, data_b3, data_a4, L3_H1, L3_W1, L3_W1, L3_W2);
    for (int j = 0; j < SIZE_A4; j++) {
      data_a4[j] = data_a4[j] / 1500;
    }
    softmax(data_a4, data_a4_exp, data_z4, SIZE_Z4);
    for (int j = 0; j < SIZE_A4; j++) {
      data_z4[j] = data_z4[j] * 1000000;
    }
    layer_3_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    dot_add_optimized(data_a0, data_w0, data_b0, data_a1, L0_H1, L0_W1, L0_W1,
                      L0_W2);
    for (int j = 0; j < SIZE_A1; j++) {
      data_a1[j] = data_a1[j] / 500;
    }
    sigmoid_optimized(data_a1, data_z1, SIZE_Z1);
    layer_0_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add_optimized(data_z1, data_w1, data_b1, data_a2, L1_H1, L1_W1, L1_W1,
                      L1_W2);
    for (int j = 0; j < SIZE_A2; j++) {
      data_a2[j] = data_a2[j] / 1500;
    }
    sigmoid_optimized(data_a2, data_z2, SIZE_Z2);
    layer_1_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add_optimized(data_z2, data_w2, data_b2, data_a3, L2_H1, L2_W1, L2_W1,
                      L2_W2);
    for (int j = 0; j < SIZE_A3; j++) {
      data_a3[j] = data_a3[j] / 1500;
    }
    sigmoid_optimized(data_a3, data_z3, SIZE_Z3);
    layer_2_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    dot_add_optimized(data_z3, data_w3, data_b3, data_a4, L3_H1, L3_W1, L3_W1,
                      L3_W2);
    for (int j = 0; j < SIZE_A4; j++) {
      data_a4[j] = data_a4[j] / 1500;
    }
    softmax_optimized(data_a4, data_a4_exp, data_z4, SIZE_Z4);
    for (int j = 0; j < SIZE_A4; j++) {
      data_z4[j] = data_z4[j] * 1000000;
    }
    layer_3_optimized_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;
  double original_total_time =
      layer_0_time + layer_1_time + layer_2_time + layer_3_time;
  double optimized_total_time = layer_0_optimized_time +
                                layer_1_optimized_time +
                                layer_2_optimized_time + layer_3_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  Layer 0 time: " << layer_0_time / iterations << " seconds" << endl;
  cout << "  Layer 1 time: " << layer_1_time / iterations << " seconds" << endl;
  cout << "  Layer 2 time: " << layer_2_time / iterations << " seconds" << endl;
  cout << "  Layer 3 time: " << layer_3_time / iterations << " seconds" << endl;
  cout << "  Single iteration time: " << original_total_time / iterations
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  Layer 0 time: " << layer_0_optimized_time / iterations
       << " seconds" << endl;
  cout << "  Layer 1 time: " << layer_1_optimized_time / iterations
       << " seconds" << endl;
  cout << "  Layer 2 time: " << layer_2_optimized_time / iterations
       << " seconds" << endl;
  cout << "  Layer 3 time: " << layer_3_optimized_time / iterations
       << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / iterations
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  Layer 0: " << layer_0_time / layer_0_optimized_time << "x"
       << endl;
  cout << "  Layer 1: " << layer_1_time / layer_1_optimized_time << "x"
       << endl;
  cout << "  Layer 2: " << layer_2_time / layer_2_optimized_time << "x"
       << endl;
  cout << "  Layer 3: " << layer_3_time / layer_3_optimized_time << "x"
       << endl;
  cout << "  Total: " << original_total_time / optimized_total_time << "x"
       << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  delete[] data_a0;
  delete[] data_a1;
  delete[] data_z1;
  delete[] data_a2;
  delete[] data_z2;
  delete[] data_a3;
  delete[] data_z3;
  delete[] data_a4;
  delete[] data_a4_exp;
  delete[] data_z4;
  delete[] data_a1_opt;
  delete[] data_z1_opt;
  delete[] data_a2_opt;
  delete[] data_z2_opt;
  delete[] data_a3_opt;
  delete[] data_z3_opt;
  delete[] data_a4_opt;
  delete[] data_z4_opt;
  delete[] data_w0;
  delete[] data_w1;
  delete[] data_w2;
  delete[] data_w3;
  delete[] data_b0;
  delete[] data_b1;
  delete[] data_b2;
  delete[] data_b3;
}

int main(int argc, char *argv[]) {
  cout << "=======================================" << endl;
  cout << "Running mlp benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  string input_path;
  string output_path;
  if (argc == 3) {
    input_path = argv[1];
    output_path = argv[2];
  } else {
    printf("Usage: ./mlp_sw <input_path> <output_path>\n");
    exit(-1);
  }
  mlp_forward(input_path, output_path);
  return 0;
}