#include "cpu_impl.h"

void knn_vote_optimized(int labels[], LabelType* max_label) 
{
    knn_vote(labels, max_label);
}