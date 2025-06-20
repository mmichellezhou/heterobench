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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct Image {
  int width;
  int height;
  int channels;
  vector<unsigned char> data;

  Image(int w, int h, int c)
      : width(w), height(h), channels(c), data(w * h * c) {}
};

Image imread(const string &filename) {
  int width, height, channels;
  unsigned char *data =
      stbi_load(filename.c_str(), &width, &height, &channels, 0);
  if (!data) {
    cerr << "Error: could not load image " << filename << endl;
    exit(1);
  }
  Image img(width, height, channels);
  copy(data, data + width * height * channels, img.data.begin());
  stbi_image_free(data);
  return img;
}

Image cvtColor(const Image &src) {
  if (src.channels != 3) {
    cerr << "Error: only 3-channel BGR images are supported" << endl;
    exit(1);
  }
  Image gray(src.width, src.height, 1);
  for (int i = 0; i < src.height; ++i) {
    for (int j = 0; j < src.width; ++j) {
      int index = i * src.width + j;
      unsigned char b = src.data[3 * index];
      unsigned char g = src.data[3 * index + 1];
      unsigned char r = src.data[3 * index + 2];
      gray.data[index] =
          static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
    }
  }
  return gray;
}

void imwrite(const string &filename, const Image &img) {
  if (stbi_write_png(filename.c_str(), img.width, img.height, img.channels,
                     img.data.data(), img.width * img.channels) == 0) {
    cerr << "Error: could not write image " << filename << endl;
    exit(1);
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

int main(int argc, char **argv) {
  cout << "=======================================" << endl;
  cout << "Running sobel_filter benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  string input_image_path;
  string output_image_path;

  if (argc == 3) {
    input_image_path = argv[1];
    output_image_path = argv[2];
  } else {
    printf("Usage: ./sobel_filter <input_image> <output_image>\n");
    exit(-1);
  }

  Image image = imread(input_image_path);
  Image gray_image = cvtColor(image);

  int width = gray_image.width;
  int height = gray_image.height;
  Image sobel_image(width, height, 1);

  const uint8_t *input_image = gray_image.data.data();
  uint8_t *output_image = sobel_image.data.data();

  uint64_t image_size = height * width;

  double *sobel_x = new double[image_size];
  double *sobel_y = new double[image_size];
  double *gradient_magnitude = new double[image_size];
  double *sobel_x_optimized = new double[image_size];
  double *sobel_y_optimized = new double[image_size];
  double *gradient_magnitude_optimized = new double[image_size];

  /* Correctness tests. */
  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  sobel_filter_x(input_image, height, width, sobel_x);
  sobel_filter_y(input_image, height, width, sobel_y);
  compute_gradient_magnitude(sobel_x, sobel_y, height, width,
                             gradient_magnitude);
  cout << "Done" << endl;

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  sobel_filter_x_optimized(input_image, height, width, sobel_x_optimized);
  sobel_filter_y_optimized(input_image, height, width, sobel_y_optimized);
  compute_gradient_magnitude_optimized(sobel_x_optimized, sobel_y_optimized,
                                       height, width,
                                       gradient_magnitude_optimized);
  cout << "Done" << endl;

  // Compare the results
  cout << "Comparing original and optimized results..." << endl;
  compareResults(sobel_x, sobel_x_optimized, image_size, "sobel_filter_x");
  compareResults(sobel_y, sobel_y_optimized, image_size, "sobel_filter_y");
  compareResults(gradient_magnitude, gradient_magnitude_optimized, image_size,
                 "compute_gradient_magnitude");

  /* Performance measurement. */
  int iterations = ITERATIONS;
  cout << "Running " << iterations
       << " iterations for performance measurement..." << endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double sobel_filter_x_time = 0;
  double sobel_filter_y_time = 0;
  double gradient_magnitude_time = 0;
  double sobel_filter_x_optimized_time = 0;
  double sobel_filter_y_optimized_time = 0;
  double gradient_magnitude_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    sobel_filter_x(input_image, height, width, sobel_x);
    sobel_filter_x_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    sobel_filter_y(input_image, height, width, sobel_y);
    sobel_filter_y_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    compute_gradient_magnitude(sobel_x, sobel_y, height, width,
                               gradient_magnitude);
    gradient_magnitude_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    sobel_filter_x_optimized(input_image, height, width, sobel_x_optimized);
    sobel_filter_x_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    sobel_filter_y_optimized(input_image, height, width, sobel_y_optimized);
    sobel_filter_y_optimized_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    compute_gradient_magnitude_optimized(sobel_x_optimized, sobel_y_optimized,
                                         height, width,
                                         gradient_magnitude_optimized);
    gradient_magnitude_optimized_time += omp_get_wtime() - start_iteration_time;
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;

  double original_total_time =
      sobel_filter_x_time + sobel_filter_y_time + gradient_magnitude_time;
  double optimized_total_time = sobel_filter_x_optimized_time +
                                sobel_filter_y_optimized_time +
                                gradient_magnitude_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  sobel_filter_x time: " << sobel_filter_x_time / iterations
       << " seconds" << endl;
  cout << "  sobel_filter_y time: " << sobel_filter_y_time / iterations
       << " seconds" << endl;
  cout << "  gradient_magnitude time: " << gradient_magnitude_time / iterations
       << " seconds" << endl;
  cout << "  Single iteration time: " << original_total_time / iterations
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  sobel_filter_x time: "
       << sobel_filter_x_optimized_time / iterations << " seconds" << endl;
  cout << "  sobel_filter_y time: "
       << sobel_filter_y_optimized_time / iterations << " seconds" << endl;
  cout << "  gradient_magnitude time: "
       << gradient_magnitude_optimized_time / iterations << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / iterations
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  sobel_filter_x: "
       << sobel_filter_x_time / sobel_filter_x_optimized_time << "x" << endl;
  cout << "  sobel_filter_y: "
       << sobel_filter_y_time / sobel_filter_y_optimized_time << "x" << endl;
  cout << "  gradient_magnitude: "
       << gradient_magnitude_time / gradient_magnitude_optimized_time << "x"
       << endl;
  cout << "  Single iteration: " << original_total_time / optimized_total_time
       << "x" << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  for (int i = 0; i < height * width; ++i) {
    output_image[i] = static_cast<uint8_t>(gradient_magnitude[i]);
  }

  delete[] sobel_x;
  delete[] sobel_y;
  delete[] gradient_magnitude;

  imwrite(output_image_path, sobel_image);

  return 0;
}