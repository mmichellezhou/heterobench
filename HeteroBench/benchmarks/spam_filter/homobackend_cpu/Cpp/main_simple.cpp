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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <time.h>

// other headers
#include "cpu_impl.h"

#include "cpu_impl_optimized.h"

using namespace std;

void print_usage(char *filename) {
  printf("usage: %s <options>\n", filename);
  printf("  -f [kernel file]\n");
  printf("  -p [path to data]\n");
}

void parse_sdaccel_command_line_args(int argc, char **argv, string &kernelFile,
                                     string &path_to_data) {

  int c = 0;

  while ((c = getopt(argc, argv, "f:p:")) != -1) {
    switch (c) {
    case 'f':
      kernelFile = optarg;
      break;
    case 'p':
      path_to_data = optarg;
      break;
    default: {
      print_usage(argv[0]);
      exit(-1);
    }
    } // matching on arguments
  } // while args present
}

void parse_sdsoc_command_line_args(int argc, char **argv,
                                   string &path_to_data) {

  int c = 0;

  while ((c = getopt(argc, argv, "f:p:")) != -1) {
    switch (c) {
    case 'p':
      path_to_data = optarg;
      break;
    default: {
      print_usage(argv[0]);
      exit(-1);
    }
    } // matching on arguments
  } // while args present
}

// data structure only used in this file
typedef struct DataSet_s {
  DataType *data_points;
  LabelType *labels;
  FeatureType *parameter_vector;
  size_t num_data_points;
  size_t num_features;
} DataSet;

// sub-functions for result checking
// dot product
float dotProduct_host(FeatureType *param_vector, DataType *data_point_i,
                      const size_t num_features) {
  FeatureType result = 0.0f;

  for (int i = 0; i < num_features; i++)
    result += param_vector[i] * data_point_i[i];

  return result;
}

// predict
LabelType getPrediction(FeatureType *parameter_vector, DataType *data_point_i,
                        size_t num_features, const double treshold = 0) {
  float parameter_vector_dot_x_i =
      dotProduct_host(parameter_vector, data_point_i, num_features);
  return (parameter_vector_dot_x_i > treshold) ? 1 : 0;
}

// compute error rate
double computeErrorRate(DataSet data_set, double &cumulative_true_positive_rate,
                        double &cumulative_false_positive_rate,
                        double &cumulative_error) {

  size_t true_positives = 0, true_negatives = 0, false_positives = 0,
         false_negatives = 0;

  for (size_t i = 0; i < data_set.num_data_points; i++) {
    LabelType prediction =
        getPrediction(data_set.parameter_vector,
                      &data_set.data_points[i * data_set.num_features],
                      data_set.num_features);
    if (prediction != data_set.labels[i]) {
      if (prediction == 1)
        false_positives++;
      else
        false_negatives++;
    } else {
      if (prediction == 1)
        true_positives++;
      else
        true_negatives++;
    }
  }

  double error_rate =
      (double)(false_positives + false_negatives) / data_set.num_data_points;

  cumulative_true_positive_rate +=
      (double)true_positives / (true_positives + false_negatives);
  cumulative_false_positive_rate +=
      (double)false_positives / (false_positives + true_negatives);
  cumulative_error += error_rate;

  return error_rate;
}

// check results
void check_results(FeatureType *param_vector, DataType *data_points,
                   LabelType *labels) {
  ofstream ofile;
  ofile.open("output.txt");
  if (ofile.is_open()) {
    ofile << "\nmain parameter vector: \n";
    for (int i = 0; i < 30; i++)
      ofile << "m[" << i << "]: " << param_vector[i] << " | ";
    ofile << endl;

    // Initialize benchmark variables
    double training_tpr = 0.0;
    double training_fpr = 0.0;
    double training_error = 0.0;
    double testing_tpr = 0.0;
    double testing_fpr = 0.0;
    double testing_error = 0.0;

    // Get Training error
    DataSet training_set;
    training_set.data_points = data_points;
    training_set.labels = labels;
    training_set.num_data_points = NUM_TRAINING;
    training_set.num_features = NUM_FEATURES;
    training_set.parameter_vector = param_vector;
    computeErrorRate(training_set, training_tpr, training_fpr, training_error);

    // Get Testing error
    DataSet testing_set;
    testing_set.data_points = &data_points[NUM_FEATURES * NUM_TRAINING];
    testing_set.labels = &labels[NUM_TRAINING];
    testing_set.num_data_points = NUM_TESTING;
    testing_set.num_features = NUM_FEATURES;
    testing_set.parameter_vector = param_vector;
    computeErrorRate(testing_set, testing_tpr, testing_fpr, testing_error);

    training_tpr *= 100.0;
    training_fpr *= 100.0;
    training_error *= 100.0;
    testing_tpr *= 100.0;
    testing_fpr *= 100.0;
    testing_error *= 100.0;

    ofile << "Training TPR: " << training_tpr << endl;
    ofile << "Training FPR: " << training_fpr << endl;
    ofile << "Training Error: " << training_error << endl;
    ofile << "Testing TPR: " << testing_tpr << endl;
    ofile << "Testing FPR: " << testing_fpr << endl;
    ofile << "Testing Error: " << testing_error << endl;
  } else {
    cout << "Failed to create output file!" << endl;
  }
}

// Compare original and optimized results
void compareResults(FeatureType *original, FeatureType *optimized, size_t size,
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
    cout << "First 10 elements of original parameter vector:" << endl;
    for (int i = 0; i < 10 && i < size; i++) {
      cout << "original[" << i << "]: " << original[i] << endl;
    }

    // print the first 10 elements of optimized results
    cout << "First 10 elements of optimized parameter vector:" << endl;
    for (int i = 0; i < 10 && i < size; i++) {
      cout << "optimized[" << i << "]: " << optimized[i] << endl;
    }
  }
}

int main(int argc, char **argv) {
  cout << "=======================================" << endl;
  cout << "Running spam_filter benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  setbuf(stdout, NULL);

  // parse command line arguments
  string path_to_data("");
  // sdaccel version and sdsoc/sw version have different command line options
  parse_sdsoc_command_line_args(argc, argv, path_to_data);

  // allocate space
  // for software verification
  DataType *data_points = new DataType[DATA_SET_SIZE];
  LabelType *labels = new LabelType[NUM_SAMPLES];
  FeatureType *param_vector = new FeatureType[NUM_FEATURES];
  FeatureType *param_vector_optimized = new FeatureType[NUM_FEATURES];

  // read in dataset
  string str_points_filepath = path_to_data + "/shuffledfeats.dat";
  string str_labels_filepath = path_to_data + "/shuffledlabels.dat";

  FILE *data_file;
  FILE *label_file;

  data_file = fopen(str_points_filepath.c_str(), "r");
  if (!data_file) {
    printf("Failed to open data file %s!\n", str_points_filepath.c_str());
    return EXIT_FAILURE;
  }
  for (int i = 0; i < DATA_SET_SIZE; i++) {
    float tmp;
    fscanf(data_file, "%f", &tmp);
    data_points[i] = tmp;
  }
  fclose(data_file);

  label_file = fopen(str_labels_filepath.c_str(), "r");
  if (!label_file) {
    printf("Failed to open label file %s!\n", str_labels_filepath.c_str());
    return EXIT_FAILURE;
  }
  for (int i = 0; i < NUM_SAMPLES; i++) {
    int tmp;
    fscanf(label_file, "%d", &tmp);
    labels[i] = tmp;
  }
  fclose(label_file);

  // reset parameter vector
  for (size_t i = 0; i < NUM_FEATURES; i++) {
    param_vector[i] = 0;
    param_vector_optimized[i] = 0;
  }

  // intermediate variable for storing gradient
  FeatureType gradient[NUM_FEATURES];
  FeatureType gradient_optimized[NUM_FEATURES];

  /* Correctness tests. */
  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  for (int training_id = 0; training_id < NUM_TRAINING; training_id++) {
    // dot product between parameter vector and data sample
    FeatureType dot =
        dotProduct(param_vector, &data_points[NUM_FEATURES * training_id]);
    // sigmoid
    FeatureType prob = Sigmoid(dot);
    // compute gradient
    computeGradient(gradient, &data_points[NUM_FEATURES * training_id],
                    (prob - labels[training_id]));
    // update parameter vector
    updateParameter(param_vector, gradient, -STEP_SIZE);
  }
  cout << "Done" << endl;

  /*
  // Check original implementation results
  cout << "Checking original implementation results...";
  check_results(param_vector, data_points, labels);
  cout << "Done" << endl;
  */

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  for (int training_id = 0; training_id < NUM_TRAINING; training_id++) {
    // dot product between parameter vector and data sample
    FeatureType dot = dotProduct_optimized(
        param_vector_optimized, &data_points[NUM_FEATURES * training_id]);
    // sigmoid
    FeatureType prob = Sigmoid_optimized(dot);
    // compute gradient
    computeGradient_optimized(gradient_optimized,
                              &data_points[NUM_FEATURES * training_id],
                              (prob - labels[training_id]));
    // update parameter vector
    updateParameter_optimized(param_vector_optimized, gradient_optimized,
                              -STEP_SIZE);
  }
  cout << "Done" << endl;

  /*
  // Check optimized implementation results
  cout << "Checking optimized implementation results..." << endl;
  check_results(param_vector_optimized, data_points, labels);
  cout << "Done" << endl;
  */

  // Compare original and optimized results
  cout << "Comparing original and optimized results..." << endl;
  compareResults(param_vector, param_vector_optimized, NUM_FEATURES,
                 "Parameter vector");

  /* Performance measurement. */
  int iterations = ITERATIONS;
  cout << "Running " << iterations
       << " iterations for performance measurement..." << endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double dotProduct_time = 0;
  double Sigmoid_time = 0;
  double computeGradient_time = 0;
  double updateParameter_time = 0;
  double dotProduct_optimized_time = 0;
  double Sigmoid_optimized_time = 0;
  double computeGradient_optimized_time = 0;
  double updateParameter_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int epoch = 0; epoch < iterations; epoch++) {
    // in each epoch, go through each training instance in sequence
    for (int training_id = 0; training_id < NUM_TRAINING; training_id++) {
      start_iteration_time = omp_get_wtime();
      // dot product between parameter vector and data sample
      FeatureType dot =
          dotProduct(param_vector, &data_points[NUM_FEATURES * training_id]);
      dotProduct_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();
      // sigmoid
      FeatureType prob = Sigmoid(dot);
      Sigmoid_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();
      // compute gradient
      computeGradient(gradient, &data_points[NUM_FEATURES * training_id],
                      (prob - labels[training_id]));
      computeGradient_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();
      // update parameter vector
      updateParameter(param_vector, gradient, -STEP_SIZE);
      updateParameter_time += omp_get_wtime() - start_iteration_time;
    }
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int epoch = 0; epoch < iterations; epoch++) {
    // in each epoch, go through each training instance in sequence
    for (int training_id = 0; training_id < NUM_TRAINING; training_id++) {
      start_iteration_time = omp_get_wtime();
      // dot product between parameter vector and data sample
      FeatureType dot = dotProduct_optimized(
          param_vector_optimized, &data_points[NUM_FEATURES * training_id]);
      dotProduct_optimized_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();
      // sigmoid
      FeatureType prob = Sigmoid_optimized(dot);
      Sigmoid_optimized_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();
      // compute gradient
      computeGradient_optimized(gradient_optimized,
                                &data_points[NUM_FEATURES * training_id],
                                (prob - labels[training_id]));
      computeGradient_optimized_time += omp_get_wtime() - start_iteration_time;

      start_iteration_time = omp_get_wtime();
      // update parameter vector
      updateParameter_optimized(param_vector_optimized, gradient_optimized,
                                -STEP_SIZE);
      updateParameter_optimized_time += omp_get_wtime() - start_iteration_time;
    }
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;

  double original_total_time = dotProduct_time + Sigmoid_time +
                               computeGradient_time + updateParameter_time;
  double optimized_total_time =
      dotProduct_optimized_time + Sigmoid_optimized_time +
      computeGradient_optimized_time + updateParameter_optimized_time;

  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  dotProduct time: " << dotProduct_time / iterations << " seconds"
       << endl;
  cout << "  Sigmoid time: " << Sigmoid_time / iterations << " seconds" << endl;
  cout << "  computeGradient time: " << computeGradient_time / iterations
       << " seconds" << endl;
  cout << "  updateParameter time: " << updateParameter_time / iterations
       << " seconds" << endl;
  cout << "  Single iteration time: " << original_total_time / iterations
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  dotProduct time: " << dotProduct_optimized_time / iterations
       << " seconds" << endl;
  cout << "  Sigmoid time: " << Sigmoid_optimized_time / iterations
       << " seconds" << endl;
  cout << "  computeGradient time: "
       << computeGradient_optimized_time / iterations << " seconds" << endl;
  cout << "  updateParameter time: "
       << updateParameter_optimized_time / iterations << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / iterations
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  dotProduct: " << dotProduct_time / dotProduct_optimized_time << "x"
       << endl;
  cout << "  Sigmoid: " << Sigmoid_time / Sigmoid_optimized_time << "x" << endl;
  cout << "  computeGradient: "
       << computeGradient_time / computeGradient_optimized_time << "x" << endl;
  cout << "  updateParameter: "
       << updateParameter_time / updateParameter_optimized_time << "x" << endl;
  cout << "  Total: " << original_total_time / optimized_total_time << "x"
       << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  delete[] data_points;
  delete[] labels;
  delete[] param_vector;
  delete[] param_vector_optimized;

  return EXIT_SUCCESS;
}
