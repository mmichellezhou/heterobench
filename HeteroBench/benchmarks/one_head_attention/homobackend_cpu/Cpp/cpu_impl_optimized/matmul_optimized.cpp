#include "cpu_impl.h"

void matmul_optimized(double *matmul_x, double *matmul_y, double *matmul_output,
                int batch_size, int input_h, int input_w, int output_w) {
    // Initialize matmul_output to 0. This is done once for the entire output array.
    for (int i = 0; i < batch_size * input_h * output_w; i++) {
        matmul_output[i] = 0;
    }

    // Define tile sizes for cache optimization. These constants are defined within the function scope.
    // TILE_SIZE is chosen to ensure that the working set (output_tile, plus parts of x and y)
    // fits within the L1 cache (e.g., 32KB per core).
    const int TILE_SIZE = 32; 
    // UNROLL_K is the unroll factor for the innermost loop to expose Instruction-Level Parallelism (ILP).
    const int UNROLL_K = 4;   

    // Loop over batches
    for (int i = 0; i < batch_size; i++) {
        // Pre-calculate base pointers for the current batch to reduce address calculations inside loops.
        double *current_output_batch_ptr = matmul_output + i * input_h * output_w;
        double *current_x_batch_ptr = matmul_x + i * input_h * input_w;
        double *current_y_batch_ptr = matmul_y + i * input_w * output_w;

        // Tiled loops for 'j' (input_h dimension)
        for (int jj = 0; jj < input_h; jj += TILE_SIZE) {
            // Calculate the actual end boundary for the current 'j' tile
            int j_block_end = (jj + TILE_SIZE < input_h ? jj + TILE_SIZE : input_h);

            // Tiled loops for 'k' (output_w dimension)
            for (int kk = 0; kk < output_w; kk += TILE_SIZE) {
                // Calculate the actual end boundary for the current 'k' tile
                int k_block_end = (kk + TILE_SIZE < output_w ? kk + TILE_SIZE : output_w);

                // Declare a temporary tile for accumulating results.
                // This array is allocated on the stack and will hold a TILE_SIZE x TILE_SIZE block
                // of the output matrix, allowing for register-level accumulation and reducing
                // read-modify-write operations to global memory.
                double output_tile[TILE_SIZE][TILE_SIZE];

                // Initialize the temporary output_tile to 0.0 for this block.
                // This is crucial as we will accumulate into it.
                for (int j_local = 0; j_local < TILE_SIZE; ++j_local) {
                    for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
                        output_tile[j_local][k_local] = 0.0;
                    }
                }

                // Tiled loops for 'l' (input_w dimension, the reduction dimension)
                for (int ll = 0; ll < input_w; ll += TILE_SIZE) {
                    // Calculate the actual end boundary for the current 'l' tile
                    int l_block_end = (ll + TILE_SIZE < input_w ? ll + TILE_SIZE : input_w);

                    // Inner loops for 'j' within the current 'j' tile
                    for (int j = jj; j < j_block_end; j++) {
                        // Inner loops for 'l' within the current 'l' tile
                        for (int l = ll; l < l_block_end; l++) {
                            // Load x[i][j][l] once into a register for reuse across the 'k' loop.
                            double x_val = current_x_batch_ptr[j * input_w + l]; 
                            
                            // Inner loop for 'k' within the current 'k' tile, with unrolling.
                            // This loop accesses y[i][l][k] and accumulates into output_tile[j-jj][k-kk].
                            // The access pattern for y[i][l][k] is contiguous for 'k', which is cache-friendly.
                            int k = kk;
                            // Process 'k' in chunks of UNROLL_K
                            for (; k + UNROLL_K <= k_block_end; k += UNROLL_K) {
                                output_tile[j - jj][k - kk] += x_val * current_y_batch_ptr[l * output_w + k];
                                output_tile[j - jj][k - kk + 1] += x_val * current_y_batch_ptr[l * output_w + k + 1];
                                output_tile[j - jj][k - kk + 2] += x_val * current_y_batch_ptr[l * output_w + k + 2];
                                output_tile[j - jj][k - kk + 3] += x_val * current_y_batch_ptr[l * output_w + k + 3];
                            }
                            // Handle any remaining 'k' elements that don't fit into a full unrolled chunk
                            for (; k < k_block_end; k++) {
                                output_tile[j - jj][k - kk] += x_val * current_y_batch_ptr[l * output_w + k];
                            }
                        }
                    }
                }

                // After accumulating all contributions for the current output tile,
                // write the results back from the temporary output_tile to the global matmul_output array.
                for (int j = jj; j < j_block_end; j++) {
                    for (int k = kk; k < k_block_end; k++) {
                        current_output_batch_ptr[j * output_w + k] = output_tile[j - jj][k - kk];
                    }
                }
            }
        }
    }
}
