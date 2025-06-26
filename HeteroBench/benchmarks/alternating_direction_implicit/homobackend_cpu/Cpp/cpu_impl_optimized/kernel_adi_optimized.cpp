#include "cpu_impl.h"

using namespace std;

void kernel_adi_optimized(int tsteps,int n,double X[N + 0][N + 0],double A[N + 0][N + 0],double B[N + 0][N + 0])
{
  kernel_adi(tsteps, n, X, A, B);
}