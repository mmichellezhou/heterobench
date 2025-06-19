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
    if (abs(data[i] - data_golden[i]) > 1e-3) {
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

int main(int argc, char *argv[]) {
  cout << "=======================================" << endl;
  cout << "Running cnn benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  string input_path;
  string output_path;
  if (argc == 3) {
    input_path = argv[1];
    output_path = argv[2];
  } else {
    printf("Usage: ./cnn_sw <input_path> <output_path>\n");
    exit(-1);
  }

  /* Variable declaration/allocation. */
  double *input_image = new double[INPUT_SIZE_H * INPUT_SIZE_W];
  double *input_padded = new double[(INPUT_SIZE_H + 2 * CONV2D_PADDING) *
                                    (INPUT_SIZE_W + 2 * CONV2D_PADDING)];
  double *W_conv = new double[CONV_KERNEL_SIZE_H * CONV_KERNEL_SIZE_W];
  double *W_fc =
      new double[FULL_CONNECT_LAYER_SIZE_H * FULL_CONNECT_LAYER_SIZE_W];
  double *b_fc = new double[FULL_CONNECT_LAYER_SIZE_W];
  double *softmax_exp_results = new double[FULL_CONNECT_LAYER_SIZE_W];

  double *conv_output = new double[CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
  double *relu_output = new double[CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
  double *pool_output =
      new double[POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH];
  double *flattened_output = new double[FLATTENED_OUTPUT_SIZE];
  double *fc_output = new double[FULL_CONNECT_LAYER_SIZE_W];
  double *softmax_output = new double[FULL_CONNECT_LAYER_SIZE_W];

  // Allocate array for golden implementation
  double *conv_output_golden =
      new double[CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
  double *relu_output_golden =
      new double[CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH];
  double *pool_output_golden =
      new double[POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH];
  double *flattened_output_golden = new double[FLATTENED_OUTPUT_SIZE];
  double *fc_output_golden = new double[FULL_CONNECT_LAYER_SIZE_W];
  double *softmax_output_golden = new double[FULL_CONNECT_LAYER_SIZE_W];

  /* Correctness tests. */
  // Run golden implementation from python version
  // readData("../../dataset/conv_output.bin", conv_output_golden,
  // CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  // readData("../../dataset/relu_output.bin", relu_output_golden,
  // CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  // readData("../../dataset/pooled_output.bin", pool_output_golden,
  // POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH);
  // readData("../../dataset/flattened_output.bin", flattened_output_golden,
  // FLATTENED_OUTPUT_SIZE); readData("../../dataset/fc_output.bin",
  // fc_output_golden, FULL_CONNECT_LAYER_SIZE_W);
  // readData("../../dataset/softmax_output.bin", softmax_output_golden,
  // FULL_CONNECT_LAYER_SIZE_W);
  readData((output_path + "/conv_output.bin").c_str(), conv_output_golden,
           CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  readData((output_path + "/relu_output.bin").c_str(), relu_output_golden,
           CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  readData((output_path + "/pooled_output.bin").c_str(), pool_output_golden,
           POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH);
  readData((output_path + "/flattened_output.bin").c_str(),
           flattened_output_golden, FLATTENED_OUTPUT_SIZE);
  readData((output_path + "/fc_output.bin").c_str(), fc_output_golden,
           FULL_CONNECT_LAYER_SIZE_W);
  readData((output_path + "/softmax_output.bin").c_str(), softmax_output_golden,
           FULL_CONNECT_LAYER_SIZE_W);

  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  // readData("../../dataset/input_image.bin", input_image, INPUT_SIZE_H *
  // INPUT_SIZE_W); readData("../../dataset/W_conv.bin", W_conv,
  // CONV_KERNEL_SIZE_H * CONV_KERNEL_SIZE_W);
  // readData("../../dataset/W_fc.bin", W_fc, FULL_CONNECT_LAYER_SIZE_H *
  // FULL_CONNECT_LAYER_SIZE_W); readData("../../dataset/b_fc.bin", b_fc,
  // FULL_CONNECT_LAYER_SIZE_W);
  readData((input_path + "/input_image.bin").c_str(), input_image,
           INPUT_SIZE_H * INPUT_SIZE_W);
  readData((input_path + "/W_conv.bin").c_str(), W_conv,
           CONV_KERNEL_SIZE_H * CONV_KERNEL_SIZE_W);
  readData((input_path + "/W_fc.bin").c_str(), W_fc,
           FULL_CONNECT_LAYER_SIZE_H * FULL_CONNECT_LAYER_SIZE_W);
  readData((input_path + "/b_fc.bin").c_str(), b_fc, FULL_CONNECT_LAYER_SIZE_W);

  conv2d(input_image, W_conv, input_padded, CONV2D_BIAS, CONV2D_STRIDE,
         CONV2D_PADDING, INPUT_SIZE_H, INPUT_SIZE_W, CONV_KERNEL_SIZE_H,
         CONV_KERNEL_SIZE_W, conv_output);
  relu(conv_output, relu_output, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  max_pooling(relu_output, POOLING_SIZE, POOLING_STRIDE, CONV_OUTPUT_HEIGHT,
              CONV_OUTPUT_WIDTH, pool_output);
  // flattened_output = pooled_output.flatten()
  for (int i = 0; i < POOLING_OUTPUT_HEIGHT; i++) {
    for (int j = 0; j < POOLING_OUTPUT_WIDTH; j++) {
      flattened_output[i * POOLING_OUTPUT_WIDTH + j] =
          pool_output[i * POOLING_OUTPUT_WIDTH + j];
    }
  }
  // Note here the size of flattened_output is 1 x FLATTENED_OUTPUT_SIZE
  dot_add(flattened_output, W_fc, b_fc, fc_output, 1, FULL_CONNECT_LAYER_SIZE_H,
          FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_W);
  for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
    fc_output[i] = fc_output[i] / 15000;
  }
  softmax(fc_output, softmax_exp_results, softmax_output,
          FULL_CONNECT_LAYER_SIZE_W);
  for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
    softmax_output[i] = softmax_output[i] * 10000;
  }
  cout << "Done" << endl;

  // Check original implementation results
  cout << "Checking original implementation results..." << endl;
  cout << "Checking conv_output... ";
  checkResult(conv_output, conv_output_golden,
              CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  cout << "Checking relu_output... ";
  checkResult(relu_output, relu_output_golden,
              CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  cout << "Checking pool_output... ";
  checkResult(pool_output, pool_output_golden,
              POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH);
  cout << "Checking flattened_output... ";
  checkResult(flattened_output, flattened_output_golden, FLATTENED_OUTPUT_SIZE);
  cout << "Checking fc_output... ";
  checkResult(fc_output, fc_output_golden, FULL_CONNECT_LAYER_SIZE_W);
  cout << "Checking softmax_output... ";
  checkResult(softmax_output, softmax_output_golden, FULL_CONNECT_LAYER_SIZE_W);

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  conv2d_optimized(input_image, W_conv, input_padded, CONV2D_BIAS,
                   CONV2D_STRIDE, CONV2D_PADDING, INPUT_SIZE_H, INPUT_SIZE_W,
                   CONV_KERNEL_SIZE_H, CONV_KERNEL_SIZE_W, conv_output);
  relu_optimized(conv_output, relu_output,
                 CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  max_pooling_optimized(relu_output, POOLING_SIZE, POOLING_STRIDE,
                        CONV_OUTPUT_HEIGHT, CONV_OUTPUT_WIDTH, pool_output);
  // flattened_output = pooled_output.flatten()
  for (int i = 0; i < POOLING_OUTPUT_HEIGHT; i++) {
    for (int j = 0; j < POOLING_OUTPUT_WIDTH; j++) {
      flattened_output[i * POOLING_OUTPUT_WIDTH + j] =
          pool_output[i * POOLING_OUTPUT_WIDTH + j];
    }
  }
  // Note here the size of flattened_output is 1 x FLATTENED_OUTPUT_SIZE
  dot_add_optimized(flattened_output, W_fc, b_fc, fc_output, 1,
                    FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_H,
                    FULL_CONNECT_LAYER_SIZE_W);
  for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
    fc_output[i] = fc_output[i] / 15000;
  }
  softmax_optimized(fc_output, softmax_exp_results, softmax_output,
                    FULL_CONNECT_LAYER_SIZE_W);
  for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
    softmax_output[i] = softmax_output[i] * 10000;
  }
  cout << "Done" << endl;

  // Check optimized implementation results
  cout << "Checking optimized implementation results..." << endl;
  cout << "Checking conv_output... ";
  checkResult(conv_output, conv_output_golden,
              CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  cout << "Checking relu_output... ";
  checkResult(relu_output, relu_output_golden,
              CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
  cout << "Checking pool_output... ";
  checkResult(pool_output, pool_output_golden,
              POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH);
  cout << "Checking flattened_output... ";
  checkResult(flattened_output, flattened_output_golden, FLATTENED_OUTPUT_SIZE);
  cout << "Checking fc_output... ";
  checkResult(fc_output, fc_output_golden, FULL_CONNECT_LAYER_SIZE_W);
  cout << "Checking softmax_output... ";
  checkResult(softmax_output, softmax_output_golden, FULL_CONNECT_LAYER_SIZE_W);

  /* Performance measurement. */
  int iterations = ITERATIONS;
  cout << "Running " << iterations
       << " iterations for performance measurement..." << endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double conv2d_time = 0;
  double relu_time = 0;
  double max_pooling_time = 0;
  double dot_add_time = 0;
  double softmax_time = 0;
  double conv2d_optimized_time = 0;
  double relu_optimized_time = 0;
  double max_pooling_optimized_time = 0;
  double dot_add_optimized_time = 0;
  double softmax_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    conv2d(input_image, W_conv, input_padded, CONV2D_BIAS, CONV2D_STRIDE,
           CONV2D_PADDING, INPUT_SIZE_H, INPUT_SIZE_W, CONV_KERNEL_SIZE_H,
           CONV_KERNEL_SIZE_W, conv_output);
    conv2d_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    relu(conv_output, relu_output, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
    relu_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    max_pooling(relu_output, POOLING_SIZE, POOLING_STRIDE, CONV_OUTPUT_HEIGHT,
                CONV_OUTPUT_WIDTH, pool_output);
    max_pooling_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < POOLING_OUTPUT_HEIGHT; i++) {
      for (int j = 0; j < POOLING_OUTPUT_WIDTH; j++) {
        flattened_output[i * POOLING_OUTPUT_WIDTH + j] =
            pool_output[i * POOLING_OUTPUT_WIDTH + j];
      }
    }

    start_iteration_time = omp_get_wtime();
    dot_add(flattened_output, W_fc, b_fc, fc_output, 1,
            FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_H,
            FULL_CONNECT_LAYER_SIZE_W);
    dot_add_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
      fc_output[i] = fc_output[i] / 15000;
    }

    start_iteration_time = omp_get_wtime();
    softmax(fc_output, softmax_exp_results, softmax_output,
            FULL_CONNECT_LAYER_SIZE_W);
    softmax_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
      softmax_output[i] = softmax_output[i] * 10000;
    }
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    conv2d_optimized(input_image, W_conv, input_padded, CONV2D_BIAS,
                     CONV2D_STRIDE, CONV2D_PADDING, INPUT_SIZE_H, INPUT_SIZE_W,
                     CONV_KERNEL_SIZE_H, CONV_KERNEL_SIZE_W, conv_output);
    conv2d_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    relu_optimized(conv_output, relu_output,
                   CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH);
    relu_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    max_pooling_optimized(relu_output, POOLING_SIZE, POOLING_STRIDE,
                          CONV_OUTPUT_HEIGHT, CONV_OUTPUT_WIDTH, pool_output);
    max_pooling_optimized_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < POOLING_OUTPUT_HEIGHT; i++) {
      for (int j = 0; j < POOLING_OUTPUT_WIDTH; j++) {
        flattened_output[i * POOLING_OUTPUT_WIDTH + j] =
            pool_output[i * POOLING_OUTPUT_WIDTH + j];
      }
    }

    start_iteration_time = omp_get_wtime();
    dot_add_optimized(flattened_output, W_fc, b_fc, fc_output, 1,
                      FULL_CONNECT_LAYER_SIZE_H, FULL_CONNECT_LAYER_SIZE_H,
                      FULL_CONNECT_LAYER_SIZE_W);
    dot_add_optimized_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
      fc_output[i] = fc_output[i] / 15000;
    }

    start_iteration_time = omp_get_wtime();
    softmax_optimized(fc_output, softmax_exp_results, softmax_output,
                      FULL_CONNECT_LAYER_SIZE_W);
    softmax_optimized_time += omp_get_wtime() - start_iteration_time;

    for (int i = 0; i < FULL_CONNECT_LAYER_SIZE_W; i++) {
      softmax_output[i] = softmax_output[i] * 10000;
    }
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;

  double original_total_time =
      conv2d_time + relu_time + max_pooling_time + dot_add_time + softmax_time;
  double optimized_total_time = conv2d_optimized_time + relu_optimized_time +
                                max_pooling_optimized_time +
                                dot_add_optimized_time + softmax_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  conv2d kernel time: " << conv2d_time / iterations << " seconds"
       << endl;
  cout << "  relu kernel time: " << relu_time / iterations << " seconds"
       << endl;
  cout << "  max_pooling kernel time: " << max_pooling_time / iterations
       << " seconds" << endl;
  cout << "  dot_add kernel time: " << dot_add_time / iterations << " seconds"
       << endl;
  cout << "  softmax kernel time: " << softmax_time / iterations << " seconds"
       << endl;
  cout << "  Single iteration time: " << original_total_time / iterations
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  conv2d kernel time: " << conv2d_optimized_time / iterations
       << " seconds" << endl;
  cout << "  relu kernel time: " << relu_optimized_time / iterations
       << " seconds" << endl;
  cout << "  max_pooling kernel time: "
       << max_pooling_optimized_time / iterations << " seconds" << endl;
  cout << "  dot_add kernel time: " << dot_add_optimized_time / iterations
       << " seconds" << endl;
  cout << "  softmax kernel time: " << softmax_optimized_time / iterations
       << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / iterations
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  conv2d kernel: " << conv2d_time / conv2d_optimized_time << "x"
       << endl;
  cout << "  relu kernel: " << relu_time / relu_optimized_time << "x" << endl;
  cout << "  max_pooling kernel: "
       << max_pooling_time / max_pooling_optimized_time << "x" << endl;
  cout << "  dot_add kernel: " << dot_add_time / dot_add_optimized_time << "x"
       << endl;
  cout << "  softmax kernel: " << softmax_time / softmax_optimized_time << "x"
       << endl;
  cout << "  Total: " << original_total_time / optimized_total_time << "x"
       << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  delete[] input_image;
  delete[] input_padded;
  delete[] W_conv;
  delete[] W_fc;
  delete[] b_fc;
  delete[] conv_output;
  delete[] relu_output;
  delete[] pool_output;
  delete[] flattened_output;
  delete[] fc_output;
  delete[] softmax_output;
  delete[] softmax_exp_results;
  delete[] conv_output_golden;
  delete[] relu_output_golden;
  delete[] pool_output_golden;
  delete[] flattened_output_golden;
  delete[] fc_output_golden;
  delete[] softmax_output_golden;

  return 0;
}

// // Reset arrays for optimized version
// cout << "\nResetting arrays for optimized version..." << endl;
// memset(conv_output, 0, CONV_OUTPUT_HEIGHT * CONV_OUTPUT_WIDTH *
// sizeof(double)); memset(relu_output, 0, CONV_OUTPUT_HEIGHT *
// CONV_OUTPUT_WIDTH * sizeof(double)); memset(pool_output, 0,
// POOLING_OUTPUT_HEIGHT * POOLING_OUTPUT_WIDTH * sizeof(double));
// memset(flattened_output, 0, FLATTENED_OUTPUT_SIZE * sizeof(double));
// memset(fc_output, 0, FULL_CONNECT_LAYER_SIZE_W * sizeof(double));
// memset(softmax_output, 0, FULL_CONNECT_LAYER_SIZE_W * sizeof(double));
// memset(softmax_exp_results, 0, FULL_CONNECT_LAYER_SIZE_W * sizeof(double));