#include "cpu_impl.h"

int grid_coord_optimized(double c)
{
    grid_coord(c);
}

int grid_coord_flat_optimized(int size, double x, double y)
{
    grid_coord_flat(size, x, y);
}

void grid_init_optimized(grid_t& grid, int size)
{
    grid_init(grid, size);
}

void grid_add_optimized(grid_t& grid, particle_t* p)
{
    grid_add(grid, p);
}

void grid_add_static_optimized(linkedlist_static grad_static[gridsize2], particle_t* p)
{
    grid_add_static(grad_static, p);
}

void grid_move_optimized(particle_t& p, double size)
{
    grid_move(p, size);
}

bool grid_remove_optimized(grid_t& grid, particle_t* p, int gridCoord)
{
    grid_remove(grid, p, gridCoord);
}

void grid_clear_optimized(grid_t& grid)
{
    grid_clear(grid);
}

int grid_size_optimized(grid_t& grid)
{
    grid_size(grid);
}