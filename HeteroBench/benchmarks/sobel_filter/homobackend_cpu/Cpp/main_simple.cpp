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
#include <vector>
#include <string>
#include <iostream>

using namespace std;

struct Image {
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;

    Image(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c) {}
};

Image imread(const std::string& filename) {
    int width, height, channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Error: could not load image " << filename << std::endl;
        exit(1);
    }
    Image img(width, height, channels);
    std::copy(data, data + width * height * channels, img.data.begin());
    stbi_image_free(data);
    return img;
}

Image cvtColor(const Image& src) {
    if (src.channels != 3) {
        std::cerr << "Error: only 3-channel BGR images are supported" << std::endl;
        exit(1);
    }
    Image gray(src.width, src.height, 1);
    for (int i = 0; i < src.height; ++i) {
        for (int j = 0; j < src.width; ++j) {
            int index = i * src.width + j;
            unsigned char b = src.data[3 * index];
            unsigned char g = src.data[3 * index + 1];
            unsigned char r = src.data[3 * index + 2];
            gray.data[index] = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    return gray;
}

void imwrite(const std::string& filename, const Image& img) {
    if (stbi_write_png(filename.c_str(), img.width, img.height, img.channels, img.data.data(), img.width * img.channels) == 0) {
        std::cerr << "Error: could not write image " << filename << std::endl;
        exit(1);
    }
}

void sobel_filter(const uint8_t *input_image, int height, int width, uint8_t *output_image) {
  uint64_t image_size = height * width;

  double *sobel_x = new double[image_size];
  double *sobel_y = new double[image_size];
  double *gradient_magnitude = new double[image_size];
  
  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ..." << std::endl;
  sobel_filter_x(input_image, height, width, sobel_x);
  sobel_filter_y(input_image, height, width, sobel_y);
  compute_gradient_magnitude(sobel_x, sobel_y, height, width, gradient_magnitude);
  std::cout << "Done" << std::endl;

  // multi iterations
  int iterations = ITERATIONS;
  std::cout << "Running " << iterations << " iterations ..." << std::endl;

  double start_whole_time = omp_get_wtime();

  double start_iteration_time;
  double sobel_filter_x_time = 0;
  double sobel_filter_y_time = 0;
  double gradient_magnitude_time = 0;

  for (int i = 0; i < iterations; i++) {
    start_iteration_time = omp_get_wtime();
    sobel_filter_x(input_image, height, width, sobel_x);
    sobel_filter_x_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    sobel_filter_y(input_image, height, width, sobel_y);
    sobel_filter_y_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();
    compute_gradient_magnitude(sobel_x, sobel_y, height, width, gradient_magnitude);
    gradient_magnitude_time += omp_get_wtime() - start_iteration_time;
  }
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations " << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "Sobel filter x time: " << (sobel_filter_x_time / iterations) * 1000 << " ms" << endl;
  cout << "Sobel filter y time: " << (sobel_filter_y_time / iterations) * 1000 << " ms" << endl;
  cout << "Gradient magnitude time: " << (gradient_magnitude_time / iterations) * 1000 << " ms" << endl;

  for (int i = 0; i < height * width; ++i) {
    output_image[i] = static_cast<uint8_t>(gradient_magnitude[i]);
  }

  delete[] sobel_x;
  delete[] sobel_y;
  delete[] gradient_magnitude;
}

void sobel_filter_wrapper(string input_image_path, string output_image_path) {
    Image input_image = imread(input_image_path);
    Image gray_image = cvtColor(input_image);

    int width = gray_image.width;
    int height = gray_image.height;
    Image sobel_image(width, height, 1);

    sobel_filter(gray_image.data.data(), height, width, sobel_image.data.data());

    imwrite(output_image_path, sobel_image);
}

int main(int argc, char **argv) {
    std::cout << "=======================================" << std::endl;
    std::cout << "Running sobel_filter benchmark C++ Serial" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    string input_image_path;
    string output_image_path;

    if (argc == 3) {
        input_image_path = argv[1];
        output_image_path = argv[2];
    } else {
        printf("Usage: ./sobel_filter <input_image> <output_image>\n");
        exit(-1);
    }

    sobel_filter_wrapper(input_image_path, output_image_path);

    return 0;
}