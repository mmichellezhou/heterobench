#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

//
//  saving parameters
//
// const int NSTEPS = 20;
// const int NPARTICLES = 1000000;
// const int SAVEFREQ = 10;

//
// grid routines
//

int grid_coord_optimized(double c);
int grid_coord_flat_optimized(int size, double x, double y);
void grid_init_optimized(grid_t &grid, int size);
void grid_add_optimized(grid_t &grid, particle_t *particle);
bool grid_remove_optimized(grid_t &grid, particle_t *p, int gridCoord = -1);
void grid_move_optimized(particle_t &p, double size);
void grid_clear_optimized(grid_t &grid);
int grid_size_optimized(grid_t &grid);
void grid_add_static_optimized(linkedlist_static grad_static[gridsize2], particle_t *p);
void compute_forces_static_optimized(particle_t *particles, int n,
                                     linkedlist_static grid_static[gridsize2]);
void move_particles_static_optimized(particle_t *particles, int n, 
                                     linkedlist_static grid[gridsize2]);