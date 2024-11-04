import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
grid_size = (10, 10)
num_squares = 10
population_size = 100
generations = 100
initial_mutation_rate = 0.3
corner_target = (0, 0)

# To record the best fitness score over generations for visualization
best_fitness_scores = []
best_arrangements = []

# Fitness function with center of mass focus
def fitness(arrangement):
    distances = np.sqrt((arrangement[:, 0] - corner_target[0])**2 + (arrangement[:, 1] - corner_target[1])**2)
    center_mass = np.mean(arrangement, axis=0)
    center_dist = np.sqrt((center_mass[0] - corner_target[0])**2 + (center_mass[1] - corner_target[1])**2)
    return -np.sum(distances) - center_dist

def create_random_arrangement():
    positions = random.sample([(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])], num_squares)
    return np.array(positions)

# Genetic algorithm loop with elitism and adaptive mutation
population = [create_random_arrangement() for _ in range(population_size)]
for generation in range(generations):
    mutation_rate = initial_mutation_rate * (1 - generation / generations)  # Adaptive mutation rate

    # Calculate fitness for each arrangement
    fitness_scores = np.array([fitness(arrangement) for arrangement in population])
    best_fitness_scores.append(np.max(fitness_scores))  # Record the best fitness score of this generation

    # Selection with elitism
    sorted_indices = np.argsort(fitness_scores)
    elite_count = int(0.1 * population_size)
    selected_population = [population[i] for i in sorted_indices[-elite_count:]]
    remaining_selection = [population[i] for i in sorted_indices[-population_size//2:]]

    # Crossover and mutation
    new_population = selected_population[:]
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(remaining_selection, 2)
        child_positions = []
        
        # Uniform crossover with constraint repair
        for pos1, pos2 in zip(parent1, parent2):
            child_positions.append(pos1 if random.random() < 0.5 else pos2)
        unique_positions = set(tuple(pos) for pos in child_positions)
        while len(unique_positions) < num_squares:
            available_positions = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1]) if (i, j) not in unique_positions]
            if available_positions:  # Check if there are any positions left
                new_pos = random.choice(available_positions)
                unique_positions.add(new_pos)
            else:
                break
        child = np.array(list(unique_positions))
        
        # Mutation with adaptive rate
        if random.random() < mutation_rate:
            idx = random.randint(0, num_squares - 1)
            available_positions = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1]) if (i, j) not in child]
            if available_positions:  # Ensure there are available positions to mutate
                new_position = random.choice(available_positions)
                child[idx] = new_position

        new_population.append(child)
    
    population = new_population

    # Store the best arrangement of each generation for animation
    best_arrangement = population[np.argmax(fitness_scores)]
    best_arrangements.append(best_arrangement)

# Set up the plot for animation with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot for arrangement
scatter = ax1.scatter([], [], color='blue', s=100, marker='s')
ax1.set_xlim(-1, grid_size[1])
ax1.set_ylim(-1, grid_size[0])
ax1.invert_yaxis()
ax1.set_xlabel("X Position")
ax1.set_ylabel("Y Position")

# Line plot for fitness score evolution
line, = ax2.plot([], [], color='orange')
ax2.set_xlim(0, generations)
ax2.set_ylim(min(best_fitness_scores) - 10, max(best_fitness_scores) + 10)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Best Fitness Score")
ax2.set_title("Evolution of Best Fitness Score")

# Update function for each frame in the animation
def update(frame):
    arrangement = best_arrangements[frame]
    scatter.set_offsets(arrangement[:, [1, 0]])  # Set x and y for squares
    ax1.set_title(f"Generation {frame * 10} - Best Arrangement")  # Update title with generation

    # Update fitness score plot
    line.set_data(range(0, frame * 10 + 10, 10), best_fitness_scores[:frame + 1])

    return scatter, line

# Create the animation
ani = FuncAnimation(fig, update, frames=range(0, generations, 10), interval=500, blit=True)
plt.show()
