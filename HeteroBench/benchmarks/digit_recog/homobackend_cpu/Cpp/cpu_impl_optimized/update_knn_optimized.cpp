#include "cpu_impl.h"

void update_knn_optimized(const DigitType training_set[NUM_TRAINING * DIGIT_WIDTH], 
                const DigitType test_set[DIGIT_WIDTH],
                int dists[K_CONST], int labels[K_CONST])
{
    update_knn(training_set, test_set, dists, labels);
}