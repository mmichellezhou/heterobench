#include <iostream>

void sobel_filter_x_optimized(const uint8_t *input_image, int height, int width, double *sobel_x);
void sobel_filter_y_optimized(const uint8_t *input_image, int height, int width, double *sobel_y);
void compute_gradient_magnitude_optimized(const double *sobel_x, const double *sobel_y, int height, int width, double *gradient_magnitude);
