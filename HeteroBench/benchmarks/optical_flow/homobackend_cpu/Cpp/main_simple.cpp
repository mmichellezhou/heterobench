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

// standard C/C++ headers
#include "omp.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <time.h>

// other headers
#include "../../imageLib/imageLib.h"
#include "cpu_impl.h"

#include "cpu_impl_optimized.h"

using namespace std;

void write_results(velocity_t output[MAX_HEIGHT][MAX_WIDTH],
                   CFloatImage refFlow, string outFile) {
  // copy the output into the float image
  CFloatImage outFlow(MAX_WIDTH, MAX_HEIGHT, 2);
  for (int i = 0; i < MAX_HEIGHT; i++) {
    for (int j = 0; j < MAX_WIDTH; j++) {
      double out_x = output[i][j].x;
      double out_y = output[i][j].y;

      if (out_x * out_x + out_y * out_y > 25.0) {
        outFlow.Pixel(j, i, 0) = 1e10;
        outFlow.Pixel(j, i, 1) = 1e10;
      } else {
        outFlow.Pixel(j, i, 0) = out_x;
        outFlow.Pixel(j, i, 1) = out_y;
      }
    }
  }

  cout << "Writing output flow file..." << endl;
  WriteFlowFile(outFlow, outFile.c_str());
  cout << "Output flow file written to " << outFile << endl;
}

void print_usage(char *filename) {
  printf("usage: %s <options>\n", filename);
  printf("  -f [kernel file]\n");
  printf("  -p [path to data]\n");
  printf("  -o [path to output]\n");
}

void parse_sdsoc_command_line_args(int argc, char **argv, string &dataPath,
                                   string &outFileOrig, string &outFileOpt) {

  int c = 0;

  while ((c = getopt(argc, argv, "p:o:O:")) != -1) {
    switch (c) {
    case 'p':
      dataPath = optarg;
      break;
    case 'o':
      outFileOrig = optarg;
      break;
    case 'O':
      outFileOpt = optarg;
      break;
    default: {
      print_usage(argv[0]);
      exit(-1);
    }
    } // matching on arguments
  } // while args present
}

void compareResults(const velocity_t out1[MAX_HEIGHT][MAX_WIDTH],
                    const velocity_t out2[MAX_HEIGHT][MAX_WIDTH],
                    double threshold, const std::string &name) {
  int error = 0;
  for (int i = 0; i < MAX_HEIGHT; ++i) {
    for (int j = 0; j < MAX_WIDTH; ++j) {
      float x1 = out1[i][j].x;
      float y1 = out1[i][j].y;
      float x2 = out2[i][j].x;
      float y2 = out2[i][j].y;
      double dx = x1 - x2;
      double dy = y1 - y2;
      double epe = sqrt(dx * dx + dy * dy);
      if (epe > threshold) {
        error++;
      }
    }
  }
  if (!error) {
    cout << name << ": Pass (Original and optimized match)" << endl;
  } else {
    cout << name << ": Fail (Original and optimized differ)" << endl;
    cout << "error: " << error << " differences found" << endl;

    // print the first 10 elements of original results
    cout << "First 10 elements of original results: ";
    int count = 0;
    for (int i = 0; i < MAX_HEIGHT && count < 10; ++i) {
      for (int j = 0; j < MAX_WIDTH && count < 10; ++j) {
        cout << "(" << out1[i][j].x << ", " << out1[i][j].y << ") ";
        count++;
      }
    }
    cout << endl;

    // print the first 10 elements of optimized results
    cout << "First 10 elements of optimized results: ";
    count = 0;
    for (int i = 0; i < MAX_HEIGHT && count < 10; ++i) {
      for (int j = 0; j < MAX_WIDTH && count < 10; ++j) {
        cout << "(" << out2[i][j].x << ", " << out2[i][j].y << ") ";
        count++;
      }
    }
    cout << endl;
  }
}

int main(int argc, char **argv) {
  cout << "=======================================" << endl;
  cout << "Running optical_flow benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  // parse command line arguments
  string dataPath("");
  string outFileOrig("");
  string outFileOpt("");
  parse_sdsoc_command_line_args(argc, argv, dataPath, outFileOrig, outFileOpt);

  // create actual file names according to the datapath
  string frame_files[5];
  string reference_file;
  frame_files[0] = dataPath + "/frame1.ppm";
  frame_files[1] = dataPath + "/frame2.ppm";
  frame_files[2] = dataPath + "/frame3.ppm";
  frame_files[3] = dataPath + "/frame4.ppm";
  frame_files[4] = dataPath + "/frame5.ppm";
  reference_file = dataPath + "/ref.flo";

  CByteImage imgs[5];
  for (int i = 0; i < 5; i++) {
    CByteImage tmpImg;
    ReadImage(tmpImg, frame_files[i].c_str());
    imgs[i] = ConvertToGray(tmpImg);
  }

  /* Correctness tests. */
  // Run golden implementation
  CFloatImage refFlow;
  ReadFlowFile(refFlow, reference_file.c_str());

  // sw version host code
  static pixel_t frames[5][MAX_HEIGHT][MAX_WIDTH];
  static velocity_t outputs[MAX_HEIGHT][MAX_WIDTH];
  static velocity_t outputs_optimized[MAX_HEIGHT][MAX_WIDTH];

  // use native C datatype arrays
  for (int f = 0; f < 5; f++)
    for (int i = 0; i < MAX_HEIGHT; i++)
      for (int j = 0; j < MAX_WIDTH; j++)
        frames[f][i][j] = imgs[f].Pixel(j, i, 0) / 255.0f;

  // intermediate arrays
  static pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH];
  static gradient_t y_filtered[MAX_HEIGHT][MAX_WIDTH];
  static gradient_t filtered_gradient[MAX_HEIGHT][MAX_WIDTH];
  static outer_t out_product[MAX_HEIGHT][MAX_WIDTH];
  static tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH];
  static tensor_t tensor[MAX_HEIGHT][MAX_WIDTH];

  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  gradient_xy_calc(frames[2], gradient_x, gradient_y);
  gradient_z_calc(frames[0], frames[1], frames[2], frames[3], frames[4],
                  gradient_z);
  gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered);
  gradient_weight_x(y_filtered, filtered_gradient);
  outer_product(filtered_gradient, out_product);
  tensor_weight_y(out_product, tensor_y);
  tensor_weight_x(tensor_y, tensor);
  flow_calc(tensor, outputs);
  cout << "Done" << endl;

  // After original warm up run
  write_results(outputs, refFlow, outFileOrig);

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  gradient_xy_calc_optimized(frames[2], gradient_x, gradient_y);
  gradient_z_calc_optimized(frames[0], frames[1], frames[2], frames[3],
                            frames[4], gradient_z);
  gradient_weight_y_optimized(gradient_x, gradient_y, gradient_z, y_filtered);
  gradient_weight_x_optimized(y_filtered, filtered_gradient);
  outer_product_optimized(filtered_gradient, out_product);
  tensor_weight_y_optimized(out_product, tensor_y);
  tensor_weight_x_optimized(tensor_y, tensor);
  flow_calc_optimized(tensor, outputs_optimized);
  cout << "Done" << endl;

  // After optimized warm up run
  write_results(outputs_optimized, refFlow, outFileOpt);

  // Compare original and optimized results
  cout << "Comparing original and optimized results..." << endl;
  compareResults(outputs, outputs_optimized, 1e-4, "Flow");

  /* Performance measurement. */
  int iterations = ITERATIONS;
  cout << "Running " << iterations
       << " iterations for performance measurement..." << endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double gradient_xy_calc_time = 0;
  double gradient_z_calc_time = 0;
  double gradient_weight_y_time = 0;
  double gradient_weight_x_time = 0;
  double outer_product_time = 0;
  double tensor_weight_y_time = 0;
  double tensor_weight_x_time = 0;
  double flow_calc_time = 0;
  double gradient_xy_calc_optimized_time = 0;
  double gradient_z_calc_optimized_time = 0;
  double gradient_weight_y_optimized_time = 0;
  double gradient_weight_x_optimized_time = 0;
  double outer_product_optimized_time = 0;
  double tensor_weight_y_optimized_time = 0;
  double tensor_weight_x_optimized_time = 0;
  double flow_calc_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int iter = 0; iter < iterations; iter++) {
    start_iteration_time = omp_get_wtime();

    gradient_xy_calc(frames[2], gradient_x, gradient_y);
    gradient_xy_calc_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_z_calc(frames[0], frames[1], frames[2], frames[3], frames[4],
                    gradient_z);
    gradient_z_calc_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered);
    gradient_weight_y_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_weight_x(y_filtered, filtered_gradient);
    gradient_weight_x_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    outer_product(filtered_gradient, out_product);
    outer_product_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    tensor_weight_y(out_product, tensor_y);
    tensor_weight_y_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    tensor_weight_x(tensor_y, tensor);
    tensor_weight_x_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    flow_calc(tensor, outputs);
    flow_calc_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int iter = 0; iter < iterations; iter++) {
    start_iteration_time = omp_get_wtime();

    gradient_xy_calc_optimized(frames[2], gradient_x, gradient_y);
    gradient_xy_calc_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_z_calc_optimized(frames[0], frames[1], frames[2], frames[3],
                              frames[4], gradient_z);
    gradient_z_calc_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_weight_y_optimized(gradient_x, gradient_y, gradient_z, y_filtered);
    gradient_weight_y_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_weight_x_optimized(y_filtered, filtered_gradient);
    gradient_weight_x_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    outer_product_optimized(filtered_gradient, out_product);
    outer_product_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    tensor_weight_y_optimized(out_product, tensor_y);
    tensor_weight_y_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    tensor_weight_x_optimized(tensor_y, tensor);
    tensor_weight_x_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    flow_calc_optimized(tensor, outputs_optimized);
    flow_calc_optimized_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;

  double original_total_time = gradient_xy_calc_time + gradient_z_calc_time +
                               gradient_weight_y_time + gradient_weight_x_time +
                               outer_product_time + tensor_weight_y_time +
                               tensor_weight_x_time + flow_calc_time;
  double optimized_total_time =
      gradient_xy_calc_optimized_time + gradient_z_calc_optimized_time +
      gradient_weight_y_optimized_time + gradient_weight_x_optimized_time +
      outer_product_optimized_time + tensor_weight_y_optimized_time +
      tensor_weight_x_optimized_time + flow_calc_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  gradient_xy_calc time: " << gradient_xy_calc_time / iterations
       << " seconds" << endl;
  cout << "  gradient_z_calc time: " << gradient_z_calc_time / iterations
       << " seconds" << endl;
  cout << "  gradient_weight_y time: " << gradient_weight_y_time / iterations
       << " seconds" << endl;
  cout << "  gradient_weight_x time: " << gradient_weight_x_time / iterations
       << " seconds" << endl;
  cout << "  outer_product time: " << outer_product_time / iterations
       << " seconds" << endl;
  cout << "  tensor_weight_y time: " << tensor_weight_y_time / iterations
       << " seconds" << endl;
  cout << "  tensor_weight_x time: " << tensor_weight_x_time / iterations
       << " seconds" << endl;
  cout << "  flow_calc time: " << flow_calc_time / iterations << " seconds"
       << endl;
  cout << "  Single iteration time: " << original_total_time / iterations
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  gradient_xy_calc time: "
       << gradient_xy_calc_optimized_time / iterations << " seconds" << endl;
  cout << "  gradient_z_calc time: "
       << gradient_z_calc_optimized_time / iterations << " seconds" << endl;
  cout << "  gradient_weight_y time: "
       << gradient_weight_y_optimized_time / iterations << " seconds" << endl;
  cout << "  gradient_weight_x time: "
       << gradient_weight_x_optimized_time / iterations << " seconds" << endl;
  cout << "  outer_product time: " << outer_product_optimized_time / iterations
       << " seconds" << endl;
  cout << "  tensor_weight_y time: "
       << tensor_weight_y_optimized_time / iterations << " seconds" << endl;
  cout << "  tensor_weight_x time: "
       << tensor_weight_x_optimized_time / iterations << " seconds" << endl;
  cout << "  flow_calc time: " << flow_calc_optimized_time / iterations
       << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / iterations
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  gradient_xy_calc: "
       << gradient_xy_calc_time / gradient_xy_calc_optimized_time << "x"
       << endl;
  cout << "  gradient_z_calc: "
       << gradient_z_calc_time / gradient_z_calc_optimized_time << "x" << endl;
  cout << "  gradient_weight_y: "
       << gradient_weight_y_time / gradient_weight_y_optimized_time << "x"
       << endl;
  cout << "  gradient_weight_x: "
       << gradient_weight_x_time / gradient_weight_x_optimized_time << "x"
       << endl;
  cout << "  outer_product: "
       << outer_product_time / outer_product_optimized_time << "x" << endl;
  cout << "  tensor_weight_y: "
       << tensor_weight_y_time / tensor_weight_y_optimized_time << "x" << endl;
  cout << "  tensor_weight_x: "
       << tensor_weight_x_time / tensor_weight_x_optimized_time << "x" << endl;
  cout << "  flow_calc: " << flow_calc_time / flow_calc_optimized_time << "x"
       << endl;
  cout << "  Total: " << original_total_time / optimized_total_time << "x"
       << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  return EXIT_SUCCESS;
}
