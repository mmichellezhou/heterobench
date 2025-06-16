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
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <cmath>
#include <time.h>
#include <fstream>
#include <sys/time.h>
#include "omp.h"

// other headers
#include "cpu_impl.h"
#include "../../imageLib/imageLib.h"

#include "cpu_impl_optimized.h"

using namespace std;

void write_results(velocity_t output[MAX_HEIGHT][MAX_WIDTH], CFloatImage refFlow, std::string outFile)
{
  // copy the output into the float image
  CFloatImage outFlow(MAX_WIDTH, MAX_HEIGHT, 2);
  for (int i = 0; i < MAX_HEIGHT; i++) 
  {
    for (int j = 0; j < MAX_WIDTH; j++) 
    {
      double out_x = output[i][j].x;
      double out_y = output[i][j].y;

      if (out_x * out_x + out_y * out_y > 25.0) 
      {
        outFlow.Pixel(j, i, 0) = 1e10;
        outFlow.Pixel(j, i, 1) = 1e10;
      } 
      else 
      {
        outFlow.Pixel(j, i, 0) = out_x;
        outFlow.Pixel(j, i, 1) = out_y;
      }
    }
  }

  std::cout << "Writing output flow file..." << std::endl;
  WriteFlowFile(outFlow, outFile.c_str());
  std::cout << "Output flow file written to " << outFile << std::endl;

}


void print_usage(char* filename)
{
    printf("usage: %s <options>\n", filename);
    printf("  -f [kernel file]\n");
    printf("  -p [path to data]\n");
    printf("  -o [path to output]\n");
}

void parse_sdsoc_command_line_args(
    int argc,
    char** argv,
    std::string& dataPath,
    std::string& outFile  ) 
{

  int c = 0;

  while ((c = getopt(argc, argv, "p:o:")) != -1) 
  {
    switch (c) 
    {
      case 'p':
        dataPath = optarg;
        break;
      case 'o':
        outFile = optarg;
        break;
     default:
      {
        print_usage(argv[0]);
        exit(-1);
      }
    } // matching on arguments
  } // while args present
}

// top-level sw function
void optical_flow_sw(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
                     velocity_t outputs[MAX_HEIGHT][MAX_WIDTH],
                     CFloatImage refFlow,
                     std::string outFile)
{
  // intermediate arrays
  static pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH];
  static gradient_t y_filtered[MAX_HEIGHT][MAX_WIDTH];
  static gradient_t filtered_gradient[MAX_HEIGHT][MAX_WIDTH];
  static outer_t out_product[MAX_HEIGHT][MAX_WIDTH];
  static tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH];
  static tensor_t tensor[MAX_HEIGHT][MAX_WIDTH];

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration ...";
  gradient_xy_calc(frame2, gradient_x, gradient_y);
  gradient_z_calc(frame0, frame1, frame2, frame3, frame4, gradient_z);
  gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered);
  gradient_weight_x(y_filtered, filtered_gradient);
  outer_product(filtered_gradient, out_product);
  tensor_weight_y(out_product, tensor_y);
  tensor_weight_x(tensor_y, tensor);
  flow_calc(tensor, outputs);
  std::cout << "Done" << std::endl;


  // check results
  write_results(outputs, refFlow, outFile);


  // multi iterations
  int iterations = ITERATIONS;
  std::cout << "Running " << iterations << " iterations ...";

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

  for (int iter = 0; iter < iterations; iter++)
  {
    start_iteration_time = omp_get_wtime();

    gradient_xy_calc(frame2, gradient_x, gradient_y);
    gradient_xy_calc_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_z_calc(frame0, frame1, frame2, frame3, frame4, gradient_z);
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
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations" << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "gradient_xy_calc time: " << (gradient_xy_calc_time / iterations) * 1000 << " ms" << endl;
  cout << "gradient_z_calc time: " << (gradient_z_calc_time / iterations) * 1000 << " ms" << endl;
  cout << "gradient_weight_y time: " << (gradient_weight_y_time / iterations) * 1000 << " ms" << endl;
  cout << "gradient_weight_x time: " << (gradient_weight_x_time / iterations) * 1000 << " ms" << endl;
  cout << "outer_product time: " << (outer_product_time / iterations) * 1000 << " ms" << endl;
  cout << "tensor_weight_y time: " << (tensor_weight_y_time / iterations) * 1000 << " ms" << endl;
  cout << "tensor_weight_x time: " << (tensor_weight_x_time / iterations) * 1000 << " ms" << endl;
  cout << "flow_calc time: " << (flow_calc_time / iterations) * 1000 << " ms" << endl;
}

void optical_flow_sw_optimized(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
                     velocity_t outputs[MAX_HEIGHT][MAX_WIDTH],
                     CFloatImage refFlow,
                     std::string outFile)
{
  // intermediate arrays
  static pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH];
  static pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH];
  static gradient_t y_filtered[MAX_HEIGHT][MAX_WIDTH];
  static gradient_t filtered_gradient[MAX_HEIGHT][MAX_WIDTH];
  static outer_t out_product[MAX_HEIGHT][MAX_WIDTH];
  static tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH];
  static tensor_t tensor[MAX_HEIGHT][MAX_WIDTH];

  // 1 warm up iteration
  std::cout << "Running 1 warm up iteration (optimized version)...";
  gradient_xy_calc_optimized(frame2, gradient_x, gradient_y);
  gradient_z_calc_optimized(frame0, frame1, frame2, frame3, frame4, gradient_z);
  gradient_weight_y_optimized(gradient_x, gradient_y, gradient_z, y_filtered);
  gradient_weight_x_optimized(y_filtered, filtered_gradient);
  outer_product_optimized(filtered_gradient, out_product);
  tensor_weight_y_optimized(out_product, tensor_y);
  tensor_weight_x_optimized(tensor_y, tensor);
  flow_calc_optimized(tensor, outputs);
  std::cout << "Done" << std::endl;

  // check results
  write_results(outputs, refFlow, outFile);

  // multi iterations
  int iterations = ITERATIONS;
  std::cout << "Running " << iterations << " iterations (optimized version)...";

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

  for (int iter = 0; iter < iterations; iter++)
  {
    start_iteration_time = omp_get_wtime();

    gradient_xy_calc_optimized(frame2, gradient_x, gradient_y);
    gradient_xy_calc_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_z_calc_optimized(frame0, frame1, frame2, frame3, frame4, gradient_z);
    gradient_z_calc_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_weight_y_optimized(gradient_x, gradient_y, gradient_z, y_filtered);
    gradient_weight_y_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    gradient_weight_x_optimized(y_filtered, filtered_gradient);
    gradient_weight_x_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    outer_product_optimized(filtered_gradient, out_product);
    outer_product_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    tensor_weight_y_optimized(out_product, tensor_y);
    tensor_weight_y_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    tensor_weight_x_optimized(tensor_y, tensor);
    tensor_weight_x_time += omp_get_wtime() - start_iteration_time;

    start_iteration_time = omp_get_wtime();

    flow_calc_optimized(tensor, outputs);
    flow_calc_time += omp_get_wtime() - start_iteration_time;
  }
  std::cout << "Done" << std::endl;

  double run_whole_time = omp_get_wtime() - start_whole_time;
  cout << "1 warm up iteration and " << iterations << " iterations" << endl;
  cout << "Single iteration time: " << (run_whole_time / iterations) * 1000 << " ms" << endl;
  cout << "gradient_xy_calc time: " << (gradient_xy_calc_time / iterations) * 1000 << " ms" << endl;
  cout << "gradient_z_calc time: " << (gradient_z_calc_time / iterations) * 1000 << " ms" << endl;
  cout << "gradient_weight_y time: " << (gradient_weight_y_time / iterations) * 1000 << " ms" << endl;
  cout << "gradient_weight_x time: " << (gradient_weight_x_time / iterations) * 1000 << " ms" << endl;
  cout << "outer_product time: " << (outer_product_time / iterations) * 1000 << " ms" << endl;
  cout << "tensor_weight_y time: " << (tensor_weight_y_time / iterations) * 1000 << " ms" << endl;
  cout << "tensor_weight_x time: " << (tensor_weight_x_time / iterations) * 1000 << " ms" << endl;
  cout << "flow_calc time: " << (flow_calc_time / iterations) * 1000 << " ms" << endl;
}

int main(int argc, char ** argv) 
{
  std::cout << "=======================================" << std::endl;
  std::cout << "Running optical_flow benchmark C++ Serial" << std::endl;
  std::cout << "=======================================" << std::endl;

  // parse command line arguments
  std::string dataPath("");
  std::string outFile("");
  parse_sdsoc_command_line_args(argc, argv, dataPath, outFile);

  // create actual file names according to the datapath
  std::string frame_files[5];
  std::string reference_file;
  frame_files[0] = dataPath + "/frame1.ppm";
  frame_files[1] = dataPath + "/frame2.ppm";
  frame_files[2] = dataPath + "/frame3.ppm";
  frame_files[3] = dataPath + "/frame4.ppm";
  frame_files[4] = dataPath + "/frame5.ppm";
  reference_file = dataPath + "/ref.flo";

  // read in images and convert to grayscale
  printf("Reading input files ... \n");

  CByteImage imgs[5];
  for (int i = 0; i < 5; i++) 
  {
    CByteImage tmpImg;
    ReadImage(tmpImg, frame_files[i].c_str());
    imgs[i] = ConvertToGray(tmpImg);
  }

  // read in reference flow file
  printf("Reading reference output flow... \n");

  CFloatImage refFlow;
  ReadFlowFile(refFlow, reference_file.c_str());

  // sw version host code
  static pixel_t frames[5][MAX_HEIGHT][MAX_WIDTH];
  static velocity_t outputs[MAX_HEIGHT][MAX_WIDTH];

  // use native C datatype arrays
  for (int f = 0; f < 5; f ++ )
    for (int i = 0; i < MAX_HEIGHT; i ++ )
      for (int j = 0; j < MAX_WIDTH; j ++ )
        frames[f][i][j] = imgs[f].Pixel(j, i, 0) / 255.0f;

  // Run original version
  std::cout << "\nRunning original version..." << std::endl;
  double start_time = omp_get_wtime();
  optical_flow_sw(frames[0], frames[1], frames[2], frames[3], frames[4], outputs, refFlow, outFile);
  double original_time = omp_get_wtime() - start_time;

  // Run optimized version
  std::cout << "\nRunning optimized version..." << std::endl;
  start_time = omp_get_wtime();
  optical_flow_sw_optimized(frames[0], frames[1], frames[2], frames[3], frames[4], outputs, refFlow, outFile);
  double optimized_time = omp_get_wtime() - start_time;

  // Print performance results
  std::cout << "\nPerformance Results:" << std::endl;
  std::cout << "Original version time: " << original_time << " seconds" << std::endl;
  std::cout << "Optimized version time: " << optimized_time << " seconds" << std::endl;
  std::cout << "Speedup: " << original_time / optimized_time << "x" << std::endl;

  return EXIT_SUCCESS;
}
