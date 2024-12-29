from generate_environment import generate_environment
from generate_environment import UniformGridDecomposition
from genetic_algorithm import genetic_algorithm
from plotting_functions import *

# Run our genetic algorithm on our generated environment
if __name__ == '__main__':
    environment_min = np.array([0, 0])
    environment_max = np.array([100, 100])
    environment_bounds = np.array([environment_min, environment_max])
    start = np.array([5, 5])
    end = np.array([95, 95])

    obstacle_list = generate_environment(start, end, environment_bounds, 100, 1, 2)

    # Create our uniform spacial decomposition to map obstacles to cells in our grid
    obstacle_grid = UniformGridDecomposition(environment_bounds, cell_size=10)
    for obstacle in obstacle_list:
        obstacle_grid.add_obstacle_to_grid(obstacle)

    print("Running Genetic Algorithm ... ")
    path, _ , _= genetic_algorithm(start, end, obstacle_grid, environment_bounds)
    plot_path(start, end, environment_bounds, obstacle_list, path)