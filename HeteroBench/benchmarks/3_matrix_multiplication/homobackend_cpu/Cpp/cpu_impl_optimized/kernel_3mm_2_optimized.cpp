#include "cpu_impl.h"

using namespace std;

void kernel_3mm_2_optimized(double E[NI + 0][NJ + 0], double F[NJ + 0][NL + 0], double G[NI + 0][NL + 0])
{
  kernel_3mm_2(E, F, G);
}
