import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# GA Parameters
population_size = 50
iterations = 500
plane_width = 24
plane_height = 20
N = 10  # Number of 1s in the binary matrix
initial_mutation_rate = 0.05
max_mutation_rate = 0.5
min_mutation_rate = 0.001
mutation_increase_factor = 1.2  # How much to increase mutation if progress is slow
mutation_decrease_factor = 0.8  # How much to decrease mutation if progress is fast
improvement_threshold = 0.01  # Threshold to determine if progress is slow

# Fitness evaluation function
def evaluate(individual):
    rows, cols = np.indices(individual.shape)
    return np.sum((rows + cols) * individual)

# Selection: Tournament selection
def selection(population, fitness_scores, k=3):
    selected = np.random.choice(np.arange(population_size), k, replace=False)
    selected_fitness = [fitness_scores[i] for i in selected]
    return population[selected[np.argmin(selected_fitness)]]

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, plane_width * plane_height)
    child = np.zeros_like(parent1)
    child.ravel()[:crossover_point] = parent1.ravel()[:crossover_point]
    child.ravel()[crossover_point:] = parent2.ravel()[crossover_point:]
    
    # Ensure exactly N ones in the child
    if np.sum(child) != N:
        one_indices = np.argpartition(child.ravel(), -N)[-N:]  # Get indices of largest N values
        child.ravel()[:] = 0
        child.ravel()[one_indices] = 1
    
    return child

# Mutation with dynamic mutation rate and N ones constraint
def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        one_indices = np.argwhere(individual == 1)
        zero_indices = np.argwhere(individual == 0)
        
        # Swap one random 1 with a 0
        if one_indices.size > 0 and zero_indices.size > 0:
            one_idx = random.choice(one_indices)
            zero_idx = random.choice(zero_indices)
            individual[tuple(one_idx)] = 0
            individual[tuple(zero_idx)] = 1

# GA Function with dynamic mutation rate
def ga():
    mutation_rate = initial_mutation_rate  # Start with the initial mutation rate
    # Initialize population with exactly N ones per individual
    population = np.zeros((population_size, plane_height, plane_width), dtype=int)
    for individual in population:
        one_indices = np.random.choice(plane_height * plane_width, N, replace=False)
        individual.ravel()[one_indices] = 1
    
    best_individuals = []
    best_fitness_scores = []
    mutation_rates = []  # To store mutation rate for each generation

    best_individual = None
    best_fitness = np.inf

    previous_best_fitness = np.inf  # Store the fitness from the previous generation

    for iter_num in range(iterations):
        # Evaluate fitness
        fitness_scores = np.array([evaluate(individual) for individual in population])
        
        # Find the best individual
        current_best_fitness = np.min(fitness_scores)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = np.copy(population[np.argmin(fitness_scores)])
        
        # Store the best individual, fitness score, and mutation rate for this generation
        best_individuals.append(np.copy(best_individual))
        best_fitness_scores.append(best_fitness)
        mutation_rates.append(mutation_rate)
        
        # Check how much the best fitness improved
        fitness_change = previous_best_fitness - best_fitness
        if fitness_change < improvement_threshold:
            # If improvement is slow, increase mutation rate (exploration)
            mutation_rate = min(mutation_rate * mutation_increase_factor, max_mutation_rate)
        else:
            # If improvement is rapid, decrease mutation rate (exploitation)
            mutation_rate = max(mutation_rate * mutation_decrease_factor, min_mutation_rate)
        
        # Update the previous best fitness for the next iteration
        previous_best_fitness = best_fitness

        # Generate next generation
        next_population = []
        for _ in range(population_size // 2):
            # Selection
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            
            # Crossover
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            
            # Mutation
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            
            next_population.extend([child1, child2])
        
        population = np.array(next_population)
    
    return best_individuals, best_fitness_scores, mutation_rates

# Function to animate the best individuals, displaying fitness and mutation rates
def animate_evolution(best_individuals, best_fitness_scores, mutation_rates):
    fig, ax = plt.subplots(figsize=(5, 3))
    
    def update(frame):
        ax.clear()
        ax.imshow(best_individuals[frame], cmap='binary', interpolation='nearest')
        ax.set_title(f"Generation {frame + 1}")
        ax.text(0.95, 0.01, f"Fitness: {best_fitness_scores[frame]:.4f}\nMutation Rate: {mutation_rates[frame]:.4f}",
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    ani = animation.FuncAnimation(fig, update, frames=len(best_individuals), repeat=False)
    plt.show()

# Run the GA
best_individuals, best_fitness_scores, mutation_rates = ga()

# Animate the evolution of the best individuals with fitness and mutation rates
animate_evolution(best_individuals, best_fitness_scores, mutation_rates)
