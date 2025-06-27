#include "cpu_impl.h"

using namespace std;

void kernel_3mm_1_optimized(double C[NJ + 0][NM + 0], double D[NM + 0][NL + 0], double F[NJ + 0][NL + 0])
{
  kernel_3mm_1(C, D, F);
}
