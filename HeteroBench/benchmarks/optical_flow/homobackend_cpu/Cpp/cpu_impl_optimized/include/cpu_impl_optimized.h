#ifndef __CPU_IMPL_OPTIMIZED_H__
#define __CPU_IMPL_OPTIMIZED_H__

typedef float pixel_t;
typedef float outer_pixel_t;
typedef double calc_pixel_t;
typedef float vel_pixel_t;

void gradient_xy_calc_optimized(pixel_t frame[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
    pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH]);

void gradient_z_calc_optimized(pixel_t frame0[MAX_HEIGHT][MAX_WIDTH], 
                     pixel_t frame1[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame2[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame3[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t frame4[MAX_HEIGHT][MAX_WIDTH],
                     pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH]);

void gradient_weight_y_optimized(pixel_t gradient_x[MAX_HEIGHT][MAX_WIDTH],
                       pixel_t gradient_y[MAX_HEIGHT][MAX_WIDTH],
                       pixel_t gradient_z[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH]);

void gradient_weight_x_optimized(gradient_t y_filt[MAX_HEIGHT][MAX_WIDTH],
                       gradient_t filt_grad[MAX_HEIGHT][MAX_WIDTH]);

void outer_product_optimized(gradient_t gradient[MAX_HEIGHT][MAX_WIDTH],
                   outer_t outer_product[MAX_HEIGHT][MAX_WIDTH]);

void tensor_weight_y_optimized(outer_t outer[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH]);

void tensor_weight_x_optimized(tensor_t tensor_y[MAX_HEIGHT][MAX_WIDTH],
                     tensor_t tensor[MAX_HEIGHT][MAX_WIDTH]);

void flow_calc_optimized(tensor_t tensors[MAX_HEIGHT][MAX_WIDTH],
               velocity_t output[MAX_HEIGHT][MAX_WIDTH]);
#endif
