import numpy as np
import random

from generate_environment import generate_environment
from generate_environment import UniformGridDecomposition
from plotting_functions import *

"""
Definitions:
Chromosome: Will contain a given path and way points, eg [(1, 2), (3, 4)]
Population: is the set of chromosomes we have, essentially our solution set
Generation: How many times will be evolve our population
Individual: represents one chromosome, each individual represents a solution in search space for given problem.
            essentially the chromosome is what encodes the "individual" 
"""

# Genetic Algorithm Parameters
POPULATION_SIZE = 500
MIN_CHROMOSOME_SIZE = 10
MAX_CHROMOSOME_SIZE = 100
MAX_GENERATIONS = 500
COLLISION_PENALTY = 10000
MUTATION_RATE = 0.4
CONVERGENCE_WINDOW = 25 
CONVERGENCE_TOLERANCE = 1.0
WAYPOINT_PERTURBATION = 2.5
NUMBER_OF_PARENTS = (POPULATION_SIZE // 2)
TRIAL_LIMIT = 15
ELITE_NUMBER = int(POPULATION_SIZE * 0.01)


def is_collision(start_segment, end_segment, obstacle_grid):
    """
    Determine is if the line segment passes through any obstacles
        Computes shortest distance between center of circles to line segments to determine collisions
    """
    nearby_obstacles = obstacle_grid.get_obstacles_in_line_segment(start_segment, end_segment)

    # Create a vector from start to end of our line segment
    line_vector = end_segment - start_segment
    line_vector_squared = np.dot(line_vector, line_vector)

    for obstacle in nearby_obstacles:
        center, radius = obstacle

        # Check our line segment is actually a line, if its a point, we can check euclidean distance directly
        if line_vector_squared == 0:
            distance = np.linalg.norm(center - start_segment)
            if distance <= radius:
                return True
            continue

        # Create a vector from the start of our segment to the center of the obstacle
        line_to_center_vector = center - start_segment

        # Project the line to center vector onto our line vector
        scalar_projection = np.dot(line_to_center_vector, line_vector) / line_vector_squared

        # Check if our scalar projection lands on on the line vector 
        if scalar_projection < 0 or scalar_projection > 1:
            continue
        
        # Find the end point of our projection vector
        # This will be the closest point on our line to the center of the circle
        closest_point = start_segment + (line_vector * scalar_projection)

        # Check the distance between our closest point and center 
        # If its smaller than our radius we know indeed we pass through an obstacle
        distance = np.linalg.norm(center - closest_point)
        if distance <= radius:
            return True
        
    return False
        

def is_waypoint_valid(waypoint, obstacle_grid):
    """
    Check if the supplied waypoint is inside any of the obstacles in our environment
    """
    x, y = obstacle_grid.get_cell_coords(waypoint)
    cell = (x, y)
    nearby_obstacles = obstacle_grid.grid_hash.get(cell, [])
    
    for obstacle in nearby_obstacles:
        center, radius = obstacle
        distance = np.linalg.norm(waypoint - center)

        # Check if we are inside any of our generated obstacles
        if distance <= radius:
            return False
        
    return True

def generate_waypoint(environment_bounds, obstacle_grid):
    """
    Generate a random waypoint within our environment 
    """
    env_min = environment_bounds[0]
    env_max = environment_bounds[1]

    waypoint_created = False

    while(not waypoint_created):
        # Randomly sample inside our environment bounds
        waypoint = np.random.uniform(low=env_min, high=env_max)

        # Now check if our generated waypoint is inside any obstacles in our map
        if is_waypoint_valid(waypoint, obstacle_grid):
            waypoint_created = True
        
    return waypoint


def generate_random_path(start, end, environment_bounds, obstacle_grid):
    """
    Generate a random path within our environment
    """
    num_waypoints = np.random.randint(MIN_CHROMOSOME_SIZE, MAX_CHROMOSOME_SIZE + 1)
    path = [start]
    for i in range(num_waypoints):
        waypoint = generate_waypoint(environment_bounds, obstacle_grid)
        path.append(waypoint)

    path.append(end)
    return path


def initialize_population(start, end, environment_bounds, obstacle_grid):
    """
    Generate the initial population to evolve toward optimal
    """
    # Generate an initial population to evolve
    population = []
    for idx in range(POPULATION_SIZE):
        individual = generate_random_path(start, end, environment_bounds, obstacle_grid)
        assert np.array_equal(individual[0], start)
        assert np.array_equal(individual[-1], end)

        population.append(individual)
    
    return population


def fitness(individual, obstacle_grid):
    """
    Evaluate the fitness of the supplied individual
        - Fitness is defined by path length and a penalty due to collisions
    """
    # Calculate the length of our path
    path_length = np.sum(np.linalg.norm(np.diff(individual, axis=0), axis=1))

    # Determine number of collisions 
    # Since waypoints are guaranteed to be outside obstacles, check if line segment collide with obstacles
    collision_count = 0
    for idx in range(len(individual) - 1):
        start_segment = individual[idx]
        end_segment = individual[idx + 1]
        if (is_collision(start_segment, end_segment, obstacle_grid)):
            collision_count += 1

    fitness = path_length + (COLLISION_PENALTY * collision_count)
    return fitness


def selection(population, fitnesses):
    """
    Select parents for the next generation using rank selection
    """
    # Assign selection probabilities proportional to ranks of each individual
    ranks = len(population) - np.argsort(np.argsort(np.array(fitnesses))) 
    selection_probabilities = ranks / ranks.sum()
    
    # Select parents based on the selection probabilities defined above
    selected_indices = np.random.choice(len(population), size=NUMBER_OF_PARENTS, replace=True, p=selection_probabilities)
    parents = [population[i] for i in selected_indices]
    return parents


def crossover(parent1, parent2, environment_bounds, obstacle_grid):
    """
    Apply single point cross-over given two parents to create a child chromosome
    """
    # Choose our crossover point based on the shorter parent
    min_length = min(len(parent1), len(parent2))

    if min_length <= 3:
        # Not enough waypoints to perform crossover; return copies of parents
        return parent1.copy(), parent2.copy()

    # Choose a crossover point and create children from the swapped genetic material from both parents
    crossover_point = random.randint(1, min_length - 1)
    child1 = list(parent1[:crossover_point]) + list(parent2[crossover_point:])
    child2 = list(parent2[:crossover_point]) + list(parent1[crossover_point:])

    return child1, child2


def mutate(individual, environment_bounds, obstacle_grid):
    """
    Mutates the provided individual to explore new parts of the solution space and improve genetic diversity
     Mutation types:
        1. Perturb waypoint 
        2. Add waypoint
        3. Delete waypoint
    """
    mutated_individual = individual.copy()
    env_min = environment_bounds[0]
    env_max = environment_bounds[1]

    # Mutation #1: Perturb waypoints
    for idx in range(1, len(mutated_individual) - 1):
        if np.random.rand() < MUTATION_RATE:
            for _ in range(TRIAL_LIMIT):
                mutation_vector = np.random.uniform(-WAYPOINT_PERTURBATION, WAYPOINT_PERTURBATION, size=2)
                perturbed_waypoint = mutated_individual[idx] + mutation_vector

                # Ensure our perturbation does not land us outside our environment
                perturbed_waypoint = np.clip(perturbed_waypoint, env_min, env_max)

                # Ensure our perturbation does not land us inside any obstacles
                if is_waypoint_valid(perturbed_waypoint, obstacle_grid):
                    mutated_individual[idx] = perturbed_waypoint
                    break

    # Mutation #2: Add waypoint
    if np.random.rand() < MUTATION_RATE and len(mutated_individual) - 1 < MAX_CHROMOSOME_SIZE:
        idx = np.random.randint(1, len(mutated_individual))
        for _ in range(TRIAL_LIMIT):
            # Create a new waypoint in between the target segment and then perturb it
            new_waypoint = (mutated_individual[idx - 1] + mutated_individual[idx]) / 2
            mutation_vector = np.random.uniform(-WAYPOINT_PERTURBATION, WAYPOINT_PERTURBATION, size=2)
            new_waypoint += mutation_vector

            # Ensure our added waypoint does not land us outside our environment
            new_waypoint = np.clip(new_waypoint, env_min, env_max)

            # Ensure our added waypoint does not land us inside any obstacles
            if is_waypoint_valid(new_waypoint, obstacle_grid):
                mutated_individual = np.insert(mutated_individual, idx, new_waypoint, axis=0)
                break

    # Mutation #3: Remove waypoint
    if np.random.rand() < MUTATION_RATE and len(mutated_individual) - 2 > MIN_CHROMOSOME_SIZE:
        # Randomly select a waypoint to remove from our chromosome
        idx = np.random.randint(1, len(mutated_individual) - 1)
        mutated_individual = np.delete(mutated_individual, idx, axis=0)

    return mutated_individual


def genetic_algorithm(start, end, obstacle_grid, environment_bounds):
    """
    Determine optimal path through provided obstacle field
    """

    # Generate an initial population of individuals
    population = initialize_population(start, end, environment_bounds, obstacle_grid)
    #plot_population(start, end, environment_bounds, obstacle_list, population, 5)

    #return

    best_fitness_history = []  # Keep track of a window previous best fitness values

    # Evaluate the fitness of our initial population
    fitness_values = [fitness(individual, obstacle_grid) for individual in population]
    best_fitness = min(fitness_values)
    best_fitness_history.append(best_fitness)
    best_fitness_overall = best_fitness

    generation = 0
    for generation in range(MAX_GENERATIONS):
        # Sort our population based on fitness
        sorted_indices = np.argsort(fitness_values)
        sorted_population = [population[i] for i in sorted_indices]
        sorted_fitnesses = [fitness_values[i] for i in sorted_indices]

        # Select our elites
        elites = sorted_population[:ELITE_NUMBER]
        elite_fitnesses = sorted_fitnesses[:ELITE_NUMBER]

        # Selection of parents from the rest of the population (excluding elites)
        non_elite_population = sorted_population[ELITE_NUMBER:]
        non_elite_fitnesses = sorted_fitnesses[ELITE_NUMBER:]
        parents = selection(non_elite_population, non_elite_fitnesses)

        # Generate new population through crossover and mutation
        offspring = []
        while len(offspring) < POPULATION_SIZE - ELITE_NUMBER:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2, environment_bounds, obstacle_grid)
            offspring.append(mutate(child1, environment_bounds, obstacle_grid))
            offspring.append(mutate(child2, environment_bounds, obstacle_grid))
            
        population = elites + offspring

        # For each chromosome in mutate group
        fitness_values = [fitness(individual, obstacle_grid) for individual in population]
        best_fitness = min(fitness_values)
        best_fitness_history.append(best_fitness)

        # Update overall best fitness
        if best_fitness < best_fitness_overall:
            best_fitness_overall = best_fitness

        print(f"Generation {generation}: Best Fitness = {best_fitness}, Overall Best Fitness = {best_fitness_overall}")

        # Check convergence to see if we can stop early, if our population is not improving 
        # over some specified window with some specified tolerance, terminate early
        if len(best_fitness_history) > CONVERGENCE_WINDOW:
            best_fitness_history.pop(0)  
            max_fitness = max(best_fitness_history)
            min_fitness = min(best_fitness_history)
            if max_fitness - min_fitness <= CONVERGENCE_TOLERANCE:
                print(f"Fitness values are within {CONVERGENCE_WINDOW} for the last {CONVERGENCE_WINDOW} generations. Terminating early.")
                break
        
    # Return best solution
    best_index = np.argmin(fitness_values)
    best_path = population[best_index]

    # Check if a valid solution is found
    if fitness_values[best_index] > 10000:
        print("No collision free solution found")

    print(f"Best path length: {fitness_values[best_index]}")
    return best_path, fitness_values[best_index], generation


# For testing purposes
if __name__ == '__main__':
    environment_min = np.array([0, 0])
    environment_max = np.array([100, 100])
    environment_bounds = np.array([environment_min, environment_max])
    start = np.array([5, 5])
    end = np.array([95, 95])

    obstacle_list = generate_environment(start, end, environment_bounds, 150, 1, 2)

    # Create our uniform spacial decomposition to map obstacles to cells in our grid
    obstacle_grid = UniformGridDecomposition(environment_bounds, cell_size=10)
    for obstacle in obstacle_list:
        obstacle_grid.add_obstacle_to_grid(obstacle)

    print("Running Genetic Algorithm ... ")
    path, _, _ = genetic_algorithm(start, end, obstacle_grid, environment_bounds)
    plot_path(start, end, environment_bounds, obstacle_list, path)
