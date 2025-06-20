#include <iostream>

// dataset information
// #define NUM_TRAINING 18000
// #define CLASS_SIZE 1800
// #define NUM_TEST 2000
// #define DIGIT_WIDTH 4

// typedefs
typedef unsigned long long DigitType;
typedef unsigned char LabelType;

// parameters
// #define K_CONST 3
// #define PARA_FACTOR 40

void popcount_optimized(DigitType diff, int *popcount_result);
void update_knn_optimized(const DigitType *training_set,
                          const DigitType *test_set, int dists[K_CONST],
                          int labels[K_CONST]);
void knn_vote_optimized(int lables[K_CONST], LabelType *max_label);
