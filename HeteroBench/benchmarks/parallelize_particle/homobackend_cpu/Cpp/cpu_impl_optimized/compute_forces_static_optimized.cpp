#include "cpu_impl.h"

using namespace std;

inline int Min(int a, int b) { return a < b ? a : b; }
inline int Max(int a, int b) { return a > b ? a : b; }

void compute_forces_static_optimized(particle_t *particles, int n,
                                     linkedlist_static grid_static[gridsize2])
{
    compute_forces_static(particles, n, grid_static);
}