#include "cpu_impl.h"

void update_optimized( const DigitType* training_set, const DigitType* test_set, int dists[K_CONST], int labels[K_CONST], int label)
{
    int dist = 0;
  
  for (int i = 0; i < DIGIT_WIDTH; i ++ )
  {
    DigitType diff = test_set[i] ^ training_set[i];
    // dist += popcount(diff);
    int popcount_result = 0;
    popcount(diff, &popcount_result);
    dist += popcount_result;
  }

  int max_dist = 0;
  int max_dist_id = K_CONST+1;

  // Find the max distance
  for ( int k = 0; k < K_CONST; ++k ) 
  {
    if ( dists[k] > max_dist ) 
    {
      max_dist = dists[k];
      max_dist_id = k;
    }
  }

  // Replace the entry with the max distance
  if ( dist < max_dist )
  {
    dists[max_dist_id] = dist;
    labels[max_dist_id] = label;
  }

  return;
}

void update_knn_optimized(const DigitType training_set[NUM_TRAINING * DIGIT_WIDTH], 
                const DigitType test_set[DIGIT_WIDTH],
                int dists[K_CONST], int labels[K_CONST])
{
    update_knn(training_set, test_set, dists, labels);
}