/*
 * (C) Copyright [2024] Hewlett Packard Enterprise Development LP
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the Software),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "cpu_impl.h"
#include "common.h"
#include "omp.h"

#include "cpu_impl_optimized.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define DEBUG 0
using namespace std;

linkedlist_static grid_static[gridsize2];
linkedlist_static grid_static_opt[gridsize2];
particle_t particles_dyn[NPARTICLES];
particle_t particles[NPARTICLES];
particle_t particles_opt[NPARTICLES];

bool compare_particles(const particle_t &p1, const particle_t &p2) {
  const double epsilon = 1e-10; // Tolerance for floating-point comparison
  return fabs(p1.x - p2.x) < epsilon && fabs(p1.y - p2.y) < epsilon &&
         fabs(p1.vx - p2.vx) < epsilon && fabs(p1.vy - p2.vy) < epsilon &&
         fabs(p1.ax - p2.ax) < epsilon && fabs(p1.ay - p2.ay) < epsilon;
}

bool compare_particle_arrays(particle_t *arr1, particle_t *arr2, int size) {
  for (int i = 0; i < size; ++i) {
    if (!compare_particles(arr1[i], arr2[i])) {
      return false;
    }
  }
  return true;
}

/*
void parallelize_particle_simulation(int n, int nsteps, int savefreq, FILE*
fsave)
{
        double size = SIZE;


        // Create a grid for optimizing the interactions
        int gridSize = GRID_SIZE; // TODO: Rounding errors?

        grid_t grid;
        grid_init(grid, gridSize);

        //Upper part no need to be reimplemented
        for (int i = 0; i < n; ++i)
        {
                grid_add(grid, &particles_dyn[i]);
        }

        for (int i = 0; i < grid.size * grid.size; i++) {
                if (grid.node_counts[i] > 1)
                        printf("Grid coordinate %d: %d nodes\n", i,
grid.node_counts[i]);
        }
        cout << "Running 1 warm up iteration ..." << endl;
        // 1 warm up iteration
        compute_forces(particles_dyn, n, grid);
        move_particles(particles_dyn, n, grid);


        cout << "Done" << endl;
        cout << "Running " << nsteps << " iterations ..." << endl;
        // Simulate a number of time steps
        double start_whole_time = omp_get_wtime();

        double start_iteration_time;
        double compute_forces_time = 0;
        double move_particles_time = 0;

        for (int step = 0; step < nsteps; step++)
        {
                // Compute forces
                start_iteration_time = omp_get_wtime();
                compute_forces(particles_dyn, n, grid);
                compute_forces_time += omp_get_wtime() -
start_iteration_time;

                // Move particles
                start_iteration_time = omp_get_wtime();
                move_particles(particles_dyn, n, grid);
                move_particles_time += omp_get_wtime() -
start_iteration_time;

                if (fsave && (step % savefreq) == 0)
                        save(fsave, n, particles);
        }
        cout << "Done" << endl;

        // double end_whole_time = omp_get_wtime() - start_whole_time;
        // cout << "Total " << nsteps << " number of steps" << endl;
        // cout << "Single iteration time: " << (end_whole_time / nsteps) * 1000
<< " ms" << endl;
        // cout << "Compute forces time: " << (compute_forces_time /
nsteps) * 1000 << " ms" << endl;
        // cout << "Move particles time: " << (move_particles_time /
nsteps) * 1000 << " ms" << endl;

        grid_clear(grid);


        if (fsave)
        {
                fclose(fsave);
        }
}
*/

//
//  benchmarking program
//
int main(int argc, char **argv) {
  cout << "=======================================" << endl;
  cout << "Running parallelize_particle benchmark C++ Serial" << endl;
  cout << "=======================================" << endl;

  char *savename = read_string(argc, argv, "-o", NULL);

  FILE *fsave = savename ? fopen(savename, "w") : NULL;

  int n = NPARTICLES;
  int nsteps = NSTEPS;
  int savefreq = SAVEFREQ;

  double size = set_size(n);
  init_particles(n, particles_dyn, particles);
  init_particles(n, particles_dyn, particles_opt);
  // parallelize_particle_simulation(n, nsteps, savefreq, fsave);

  /* Correctness tests. */
  // Warm up and test original implementation
  cout << "Running 1 warm up iteration for original implementation..." << endl;
  for (int i = 0; i < n; ++i) {
    grid_add_static(grid_static, &particles[i]);
  }
  compute_forces_static(particles, n, grid_static);
  move_particles_static(particles, n, grid_static);
  cout << "Done" << endl;

  // Warm up and test optimized implementation
  cout << "Running 1 warm up iteration for optimized implementation..." << endl;
  for (int i = 0; i < n; ++i) {
    grid_add_static(grid_static_opt, &particles_opt[i]);
  }
  compute_forces_static_optimized(particles_opt, n, grid_static_opt);
  move_particles_static_optimized(particles_opt, n, grid_static_opt);
  cout << "Done" << endl;

  bool arrays_equal =
      compare_particle_arrays(particles, particles_opt, NPARTICLES);

  cout << "Comparing original and optimized results..." << endl;
  if (arrays_equal) {
    cout << "Pass (Original and optimized match)" << endl;
  } else {
    cout << "Fail (Original and optimized differ)" << endl;
  }

  /* Performance measurement. */
  cout << "Running " << nsteps << " iterations for performance measurement..."
       << endl;

  double start_whole_time = omp_get_wtime();
  double start_iteration_time;
  double compute_forces_time = 0;
  double move_particles_time = 0;
  double compute_forces_optimized_time = 0;
  double move_particles_optimized_time = 0;

  // Run original implementation
  cout << "Running original implementation..." << endl;
  for (int step = 0; step < nsteps; step++) {
    // Compute forces
    start_iteration_time = omp_get_wtime();
    compute_forces_static(particles, n, grid_static);
    compute_forces_time += omp_get_wtime() - start_iteration_time;

    // Move particles
    start_iteration_time = omp_get_wtime();
    move_particles_static(particles, n, grid_static);
    move_particles_time += omp_get_wtime() - start_iteration_time;

    if (fsave && (step % savefreq) == 0)
      save(fsave, n, particles);
  }
  cout << "Done" << endl;

  // Run optimized implementation
  cout << "Running optimized implementation..." << endl;
  for (int step = 0; step < nsteps; step++) {
    // Compute forces
    start_iteration_time = omp_get_wtime();
    compute_forces_static_optimized(particles_opt, n, grid_static_opt);
    compute_forces_optimized_time +=
        omp_get_wtime() - start_iteration_time;

    // Move particles
    start_iteration_time = omp_get_wtime();
    move_particles_static_optimized(particles_opt, n, grid_static_opt);
    move_particles_optimized_time +=
        omp_get_wtime() - start_iteration_time;

    if (fsave && (step % savefreq) == 0)
      save(fsave, n, particles_opt);
  }
  cout << "Done" << endl;

  double whole_time = omp_get_wtime() - start_whole_time;
  double original_total_time = compute_forces_time + move_particles_time;
  double optimized_total_time = compute_forces_optimized_time + move_particles_optimized_time;

  /* Print results. */
  cout << "=======================================" << endl;
  cout << "Performance Results:" << endl;
  cout << "=======================================" << endl;
  cout << "Original Implementation:" << endl;
  cout << "  compute_forces_static time: " << compute_forces_time / nsteps
       << " seconds" << endl;
  cout << "  move_particles_static time: " << move_particles_time / nsteps
       << " seconds" << endl;
  cout << "  Single iteration time: " << original_total_time / nsteps
       << " seconds" << endl;
  cout << "Optimized Implementation:" << endl;
  cout << "  compute_forces_static time: " << compute_forces_optimized_time / nsteps
       << " seconds" << endl;
  cout << "  move_particles_static time: " << move_particles_optimized_time / nsteps
       << " seconds" << endl;
  cout << "  Single iteration time: " << optimized_total_time / nsteps
       << " seconds" << endl;
  cout << "Speedup:" << endl;
  cout << "  compute_forces_static: " << compute_forces_time / compute_forces_optimized_time << " x" << endl;
  cout << "  move_particles_static: " << move_particles_time / move_particles_optimized_time << " x" << endl;
  cout << "  Total: " << original_total_time / optimized_total_time << " x" << endl;
  cout << "Whole time: " << whole_time << " seconds" << endl;

  if (fsave) {
    fclose(fsave);
  }
  return 0;
}
