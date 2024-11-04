import numpy as np
import random
import matplotlib.pyplot as plt

# Parameters
grid_size = (10, 10)  # Grid dimensions (rows, cols)
num_squares = 10  # Number of squares to place on the grid
population_size = 100  # Number of arrangements in each generation
generations = 100  # Number of generations
mutation_rate = 0.1  # Probability of mutation
corner_target = (0, 0)  # Target corner (top-left)

# Fitness function: calculates how close squares are to the target corner
def fitness(arrangement):
    distances = np.sqrt((arrangement[:, 0] - corner_target[0])**2 + (arrangement[:, 1] - corner_target[1])**2)
    return -np.sum(distances)  # Higher fitness for closer squares

# Generate initial random population without overlapping
def create_random_arrangement():
    positions = random.sample([(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])], num_squares)
    return np.array(positions)

population = [create_random_arrangement() for _ in range(population_size)]

# Visualization function
def plot_arrangement(arrangement, generation):
    plt.figure(figsize=(5, 5))
    plt.scatter(arrangement[:, 1], arrangement[:, 0], color='blue', s=100, marker='s')  # Plot squares
    plt.gca().invert_yaxis()  # Invert y-axis to match grid layout
    plt.xlim(-1, grid_size[1])
    plt.ylim(-1, grid_size[0])
    plt.title(f'Generation {generation} - Best Arrangement')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()

# Genetic algorithm loop
for generation in range(generations):
    # Calculate fitness for each arrangement
    fitness_scores = np.array([fitness(arrangement) for arrangement in population])
    
    # Selection: choose top arrangements
    sorted_indices = np.argsort(fitness_scores)[-population_size//2:]  # Top 50%
    selected_population = [population[i] for i in sorted_indices]
    
    # Crossover: create children by mixing parents without duplicating positions
    new_population = []
    while len(new_population) < population_size:
        parent1, parent2 = random.sample(selected_population, 2)
        split = random.randint(1, num_squares - 1)
        child = np.vstack((parent1[:split], parent2[split:]))
        
        # Ensure no overlapping in the child
        unique_positions = set(tuple(pos) for pos in child)
        while len(unique_positions) < num_squares:
            new_pos = random.choice([(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])])
            if new_pos not in unique_positions:
                unique_positions.add(new_pos)
        child = np.array(list(unique_positions))
        
        new_population.append(child)
    
    # Mutation: randomly change positions without overlapping
    for arrangement in new_population:
        if random.random() < mutation_rate:
            idx = random.randint(0, num_squares - 1)
            new_position = random.choice([(i, j) for i in range(grid_size[0]) for j in range(grid_size[1]) if (i, j) not in arrangement])
            arrangement[idx] = new_position
    
    population = new_population

    # Visualize the best arrangement every 10 generations
    if generation % 10 == 0:
        best_arrangement = population[np.argmax(fitness_scores)]
        plot_arrangement(best_arrangement, generation)
