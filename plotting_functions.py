import numpy as np
import matplotlib.pyplot as plt


def plot_population(start, end, environment_bounds, obstacle_list, population, max_individuals):
    """
    Plot a population of paths within the environment.
    """
    env_min = environment_bounds[0]
    env_max = environment_bounds[1]

    plt.figure(figsize=(8, 8))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Initial Population of Paths')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(env_min[0], env_max[0])
    plt.ylim(env_min[1], env_max[1])

    # Plot obstacles
    for obstacle in obstacle_list:
        center, radius = obstacle
        circle = plt.Circle(center, radius, color='grey')
        plt.gca().add_patch(circle)

    # Plot the start and end points
    plt.plot(start[0], start[1], 'go', markersize=8, label='Start')
    plt.plot(end[0], end[1], 'ro', markersize=8, label='Goal')

    # Plot individuals
    for idx, path in enumerate(population):
        if idx > max_individuals:
            break
        path = np.array(path)
        x_coords = path[:, 0]
        y_coords = path[:, 1]
        plt.plot(x_coords, y_coords, 'bo-', label='Path', alpha=0.6)

    #plt.legend(loc='upper left')
    plt.show()


def plot_path(start, end, environment_bounds, obstacle_list, path):
    """
    Plot the initial population of paths within the environment.
    """
    env_min = environment_bounds[0]
    env_max = environment_bounds[1]

    plt.figure(figsize=(8, 8))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Path Generated')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(env_min[0], env_max[0])
    plt.ylim(env_min[1], env_max[1])

    # Plot obstacles
    for obstacle in obstacle_list:
        center, radius = obstacle
        circle = plt.Circle(center, radius, color='grey')
        plt.gca().add_patch(circle)

    # Plot the start and end points
    plt.plot(start[0], start[1], 'go', markersize=8, label='Start')
    plt.plot(end[0], end[1], 'ro', markersize=8, label='Goal')

    # Plot path
    path = np.array(path)
    x_coords = path[:, 0]
    y_coords = path[:, 1]
    plt.plot(x_coords, y_coords, 'bo-', label='Path', alpha=0.6)

    plt.legend(loc='upper left')
    plt.show()
