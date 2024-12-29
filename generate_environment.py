import numpy as np
import matplotlib.pyplot as plt


def generate_environment(start, goal, environment_bounds, obstacle_num, obstacle_min_radius, obstacle_max_radius):
    """
    Generate a random set of obstacles within the environment bounds provided
    """
    env_min = environment_bounds[0]
    env_max = environment_bounds[1]

    obstacle_list = []

    for i in range(obstacle_num):
        obstacle_placed = False
        while(not obstacle_placed):
            # Every now and then lets form a larger obstacle
            if i % 100 == 0:
                obstacle_center = np.random.uniform(env_min + obstacle_max_radius, env_max - obstacle_min_radius)
                obstacle_radius = obstacle_max_radius * 3
            elif i % 25 == 0:
                obstacle_center = np.random.uniform(env_min + obstacle_max_radius, env_max - obstacle_min_radius)
                obstacle_radius = obstacle_max_radius * 1.5
            else:
                obstacle_center = np.random.uniform(env_min + obstacle_max_radius, env_max - obstacle_min_radius)
                obstacle_radius = np.random.uniform(obstacle_min_radius, obstacle_max_radius)

            # Ensure that our obstacle does not overlap with our start and end points with some added padding
            start_distance = np.linalg.norm(obstacle_center - np.array(start))
            end_distance = np.linalg.norm(obstacle_center - np.array(goal))
            distance_padding = 2.5
            if start_distance > obstacle_radius + distance_padding  and end_distance > obstacle_radius + distance_padding :
                obstacle_list.append((obstacle_center, obstacle_radius))
                obstacle_placed = True

    return obstacle_list  


class UniformGridDecomposition:
    def __init__(self, environment_bounds, cell_size):
        """
        Initialize the grid decomposition with the provided bounds and cell size
        """
        self.cell_size = cell_size
        self.environment_min = environment_bounds[0]
        self.environment_max = environment_bounds[1]
        self.x_cells_number = int(np.ceil((self.environment_max[0] - self.environment_min[0]) / cell_size))
        self.y_cells_number = int(np.ceil((self.environment_max[1] - self.environment_min[1]) / cell_size))
        self.grid_hash = {} 
    
    def get_cell_coords(self, point):
        """
        Get the grid cell coordinates for the supplied point
        """
        x = max(0, min(int((point[0] - self.environment_min[0]) / self.cell_size), self.x_cells_number - 1))  
        y = max(0, min(int((point[1] - self.environment_min[1]) / self.cell_size), self.y_cells_number - 1))
        return x, y
    

    def add_obstacle_to_grid(self, obstacle):
        """
        Add the obstacle to the grid mapping of the relevant cells
        """
        center, radius = obstacle

        # Create a bounding box around our circle
        bottom_left_corner = center - radius
        top_right_corner = center + radius

        # Find the cell corresponding to the bounding box corners
        bottom_left_cell = self.get_cell_coords(bottom_left_corner)
        top_right_cell = self.get_cell_coords(top_right_corner)

        for x in range(bottom_left_cell[0], top_right_cell[0] + 1):
            for y in range(bottom_left_cell[1], top_right_cell[1] + 1):
                cell = (x, y)
                if cell not in self.grid_hash:
                    self.grid_hash[cell] = []
                self.grid_hash[cell].append((tuple(center), radius))
        
    def get_obstacles_in_line_segment(self, start_segment, end_segment):
        """
        Get a list of obstacles that are near the line segment using our grid definition
        """

        # Create a bounding box of the line segment
        bottom_left_cell = self.get_cell_coords([min(start_segment[0], end_segment[0]), 
                                                 min(start_segment[1], end_segment[1])])
        top_right_cell = self.get_cell_coords([max(start_segment[0], end_segment[0]), 
                                            max(start_segment[1], end_segment[1])])
        
        obstacle_list = []
        for x in range(bottom_left_cell[0], top_right_cell[0] + 1):
            for y in range(bottom_left_cell[1], top_right_cell[1] + 1):
                cell = (x, y)
                if cell in self.grid_hash: 
                    obstacle_list.extend(self.grid_hash[cell])

        # Return list of all unique obstacles that reside near the line segment
        return list(set(obstacle_list))


def plot_environment(start, goal, environment_bounds, obstacle_list):
    """
    Plot our generated environment for visualization
    """
    env_min = environment_bounds[0]
    env_max = environment_bounds[1]

    # Create and configure plot
    plt.figure(figsize=(6, 6))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
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
    plt.plot(start[0], start[1], 'go', markersize=5, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=5, label='Goal')
    plt.title("Sample Environment")
    plt.legend(loc='upper left')
    plt.show()

# For testing purposes
if __name__ == '__main__':
    environment_min = np.array([0, 0])
    environment_max = np.array([100, 100])
    environment_bounds = np.array([environment_min, environment_max])
    start = (5, 5)
    end = (95, 95)
    obstacle_list = generate_environment(start, end, environment_bounds, 125, 1, 2.5)
    plot_environment(start, end, environment_bounds, obstacle_list)