#ifndef __CPU_IMPL_OPTIMIZED_H__
#define __CPU_IMPL_OPTIMIZED_H__
#include <iostream>
#define KERNEL_SIZE 1
#define OFFSET 1

void gaussian_filter_optimized(const uint8_t *inImage, int height, int width,
                               uint8_t *outImage);
void gradient_intensity_direction_optimized(const uint8_t *inImage, int height,
                                            int width, double *intensity,
                                            uint8_t *direction);
void edge_thinning_optimized(double *intensity, uint8_t *direction, int height,
                             int width, double *outImage);
void double_thresholding_optimized(double *suppressed_image, int height,
                                   int width, int high_threshold,
                                   int low_threshold, uint8_t *outImage);
void hysteresis_optimized(uint8_t *inImage, int height, int width,
                          uint8_t *outImage);

#endif