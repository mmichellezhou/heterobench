#include "cpu_impl.h"

void update_knn_optimized(const DigitType training_set[NUM_TRAINING * DIGIT_WIDTH], 
                const DigitType test_set[DIGIT_WIDTH],
                int dists[K_CONST], int labels[K_CONST])
{
  // Optimize memory access: Use a pointer that increments by DIGIT_WIDTH
  // instead of recalculating the offset `i * DIGIT_WIDTH` in each iteration.
  const DigitType* current_training_digit_ptr = training_set;

  for (int i = 0; i < NUM_TRAINING; ++i) 
  {
    // Calculate the label for the current training set.
    // Modern compilers are efficient at optimizing constant integer division.
    int label = i / CLASS_SIZE; 

    // Calculate the Hamming distance between the test_set and the current training_set entry.
    int dist = 0;
    
    // Loop for calculating Hamming distance (DIGIT_WIDTH loop).
    // Apply loop unrolling (by 4) to increase instruction-level parallelism
    // and reduce loop overhead. This assumes DIGIT_WIDTH is a compile-time constant.
    // The remainder loop handles cases where DIGIT_WIDTH is not a multiple of 4.
    for (int j = 0; j < DIGIT_WIDTH; j += 4) 
    {
      // Check if there are at least 4 elements remaining to process in this unrolled block.
      if (j + 3 < DIGIT_WIDTH) {
        // Process 4 elements.
        // Use __builtin_popcount for efficient bit counting, which maps to a single
        // POPCNT instruction on supported architectures (like Intel Xeon Gold).
        dist += __builtin_popcount(test_set[j] ^ current_training_digit_ptr[j]);
        dist += __builtin_popcount(test_set[j+1] ^ current_training_digit_ptr[j+1]);
        dist += __builtin_popcount(test_set[j+2] ^ current_training_digit_ptr[j+2]);
        dist += __builtin_popcount(test_set[j+3] ^ current_training_digit_ptr[j+3]);
      } else {
        // Handle the remaining elements (0 to 3 elements) if DIGIT_WIDTH is not
        // a multiple of the unroll factor.
        for (int rem_j = j; rem_j < DIGIT_WIDTH; ++rem_j) {
          dist += __builtin_popcount(test_set[rem_j] ^ current_training_digit_ptr[rem_j]);
        }
        break; // Exit the unrolled loop after processing the remainder.
      }
    }

    // Find the maximum distance in the 'dists' array and its corresponding index.
    int max_dist = 0;
    // Initialize max_dist_id to an invalid index (K_CONST).
    // If K_CONST is 0, this value will remain, and no update will occur later.
    // If K_CONST > 0, it will be updated by the loop.
    int max_dist_id = K_CONST; 

    // Loop to find the maximum distance (K_CONST loop).
    // Apply loop unrolling (by 2) to improve instruction-level parallelism.
    // The remainder loop handles cases where K_CONST is not a multiple of 2.
    for (int k = 0; k < K_CONST; k += 2) 
    {
      // Check if there are at least 2 elements remaining to process in this unrolled block.
      if (k + 1 < K_CONST) {
        // Process two elements. If both are new maximums, the latter one's index is stored.
        // This maintains functional equivalence with the original code's behavior
        // (picking the last encountered maximum).
        if (dists[k] > max_dist) 
        {
          max_dist = dists[k];
          max_dist_id = k;
        }
        if (dists[k+1] > max_dist) 
        {
          max_dist = dists[k+1];
          max_dist_id = k+1;
        }
      } else {
        // Handle the remaining element (0 or 1 element).
        for (int rem_k = k; rem_k < K_CONST; ++rem_k) {
          if (dists[rem_k] > max_dist) 
          {
            max_dist = dists[rem_k];
            max_dist_id = rem_k;
          }
        }
        break; // Exit the unrolled loop after processing the remainder.
      }
    }

    // Replace the entry with the maximum distance if the current 'dist' is smaller.
    // If K_CONST is 0, max_dist remains 0, and 'dist < 0' is always false for non-negative distances,
    // so no update occurs, which is correct.
    if (dist < max_dist)
    {
      dists[max_dist_id] = dist;
      labels[max_dist_id] = label;
    }

    // Advance the pointer to the next training digit block for the next iteration.
    current_training_digit_ptr += DIGIT_WIDTH;
  }
}