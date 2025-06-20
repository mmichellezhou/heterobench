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

void check_results(LabelType *result, const LabelType *expected, int cnt) {
  int correct_cnt = 0;

  std::ofstream ofile;
  ofile.open("outputs.txt");
  if (ofile.is_open()) {
    for (int i = 0; i < cnt; i++) {
      if (result[i] != expected[i])
        ofile << "Test " << i << ": expected = " << int(expected[i])
              << ", result = " << int(result[i]) << endl;
      else
        correct_cnt++;
    }

    ofile << "\n\t " << correct_cnt << " / " << cnt << " correct!" << endl;
    ofile.close();
  } else {
    cout << "Failed to create output file!" << endl;
  }
}

const DigitType testing_data[NUM_TEST * DIGIT_WIDTH] = {
#include "../../196data/test_set.dat"
};

const LabelType expected[NUM_TEST] = {
#include "../../196data/expected.dat"
};

const DigitType training_data[NUM_TRAINING * DIGIT_WIDTH] = {
#include "../../196data/training_set_0.dat"
#include "../../196data/training_set_1.dat"
#include "../../196data/training_set_2.dat"
#include "../../196data/training_set_3.dat"
#include "../../196data/training_set_4.dat"
#include "../../196data/training_set_5.dat"
#include "../../196data/training_set_6.dat"
#include "../../196data/training_set_7.dat"
#include "../../196data/training_set_8.dat"
#include "../../196data/training_set_9.dat"
};

/* Compare original and optimized results */
void compareResults(LabelType *original, LabelType *optimized, int size,
                    const string &name) {
  int error = 0;
  for (int i = 0; i < size; i++) {
    if (original[i] != optimized[i]) {
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
      cout << (int)original[i] << " ";
    }
    cout << endl;

    // print the first 10 elements of optimized results
    cout << "First 10 elements of optimized results: ";
    for (int i = 0; i < 10 && i < size; i++) {
      cout << (int)optimized[i] << " ";
    }
    cout << endl;
  }
}

int main(int argc, char **argv) {
  cout << "=======================================" << endl;
  cout << "Running digit_recog benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  // sw version host code
  // create space for the result
  LabelType *result = new LabelType[NUM_TEST];
  LabelType *result_optimized = new LabelType[NUM_TEST];

  // Initialize arrays to 0 to prevent comparison of uninitialized values
  for (int i = 0; i < NUM_TEST; i++) {
    result[i] = 0;
    result_optimized[i] = 0;
  }

  // nearest neighbor set
  int dists[K_CONST];
  int labels[K_CONST];

  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  for (int i = 0; i < K_CONST; ++i) {
    dists[i] = 256;
    labels[i] = 0;
  }
  update_knn(training_data, &testing_data[0], dists, labels);
  LabelType max_label = 0;
  knn_vote(labels, &max_label);
  result[0] = max_label;
  cout << "Done" << endl;

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  for (int i = 0; i < K_CONST; ++i) {
    dists[i] = 256;
    labels[i] = 0;
  }
  update_knn_optimized(training_data, &testing_data[0], dists, labels);
  knn_vote_optimized(labels, &max_label);
  result_optimized[0] = max_label;
  cout << "Done" << endl;

  // Compare results
  cout << "Comparing original and optimized results..." << endl;
  compareResults(result, result_optimized, NUM_TEST, "Digit recognition");

  /* Performance measurement. */
  cout << "Running " << NUM_TEST << " iterations for performance measurement..."
       << endl;
  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double update_knn_time = 0;
  double knn_vote_time = 0;
  double update_knn_optimized_time = 0;
  double knn_vote_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  // loop through test set
  for (int t = 0; t < NUM_TEST; ++t) {
    // Initialize the neighbor set
    for (int i = 0; i < K_CONST; ++i) {
      // Note that the max distance is 256
      dists[i] = 256;
      labels[i] = 0;
    }

    // for each training instance, compare it with the test instance, and update
    // the nearest neighbor set
    start_iteration_time = omp_get_wtime();
    update_knn(training_data, &testing_data[t * DIGIT_WIDTH], dists, labels);
    update_knn_time += omp_get_wtime() - start_iteration_time;

    // Compute the final output
    start_iteration_time = omp_get_wtime();
    LabelType max_label = 0;
    knn_vote(labels, &max_label);
    result[t] = max_label;
    knn_vote_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  /*
  // Check original implementation results
  cout << "Checking original implementation results..." << endl;
  check_results(result, expected, NUM_TEST);
  cout << "Done" << endl;
  */

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  // loop through test set
  for (int t = 0; t < NUM_TEST; ++t) {
    // Initialize the neighbor set
    for (int i = 0; i < K_CONST; ++i) {
      // Note that the max distance is 256
      dists[i] = 256;
      labels[i] = 0;
    }

    // for each training instance, compare it with the test instance, and update
    // the nearest neighbor set
    start_iteration_time = omp_get_wtime();
    update_knn_optimized(training_data, &testing_data[t * DIGIT_WIDTH], dists,
                         labels);
    update_knn_optimized_time += omp_get_wtime() - start_iteration_time;

    // Compute the final output
    start_iteration_time = omp_get_wtime();
    LabelType max_label = 0;
    knn_vote_optimized(labels, &max_label);
    result_optimized[t] = max_label;
    knn_vote_optimized_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  /*
  // Check optimized implementation results
  cout << "Checking optimized implementation results..." << endl;
  check_results(result_optimized, expected, NUM_TEST);
  cout << "Done" << endl;
  */

  double whole_time = omp_get_wtime() - start_whole_time;

  double original_total_time = update_knn_time + knn_vote_time;
  double optimized_total_time =
      update_knn_optimized_time + knn_vote_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  update_knn time: " << update_knn_time / NUM_TEST << " seconds"
       << endl;
  cout << "  knn_vote time: " << knn_vote_time / NUM_TEST << " seconds" << endl;
  cout << "  Single iteration time: " << original_total_time / NUM_TEST
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  update_knn time: " << update_knn_optimized_time / NUM_TEST
       << " seconds" << endl;
  cout << "  knn_vote time: " << knn_vote_optimized_time / NUM_TEST
       << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / NUM_TEST
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  update_knn: " << update_knn_time / update_knn_optimized_time << "x"
       << endl;
  cout << "  knn_vote: " << knn_vote_time / knn_vote_optimized_time << "x"
       << endl;
  cout << "  Total: " << original_total_time / optimized_total_time << "x"
       << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  delete[] result;
  delete[] result_optimized;

  return EXIT_SUCCESS;
}
