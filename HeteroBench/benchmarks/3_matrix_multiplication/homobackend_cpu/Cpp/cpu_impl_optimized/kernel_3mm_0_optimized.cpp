#include "cpu_impl.h"

using namespace std;

void kernel_3mm_0_optimized(double A[NI + 0][NK + 0], double B[NK + 0][NJ + 0], double E[NI + 0][NJ + 0])
{
  kernel_3mm_0(A, B, E);
}
