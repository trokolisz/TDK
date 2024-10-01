import numpy as np
import random
import matplotlib.pyplot as plt


num_particles = 50
iterations = 1000

N = 10

plane_width = 24
plane_height = 20


def visualize_matrix(matrix, title="Binary Matrix"):
    plt.figure(figsize=(5, 3))
    plt.imshow(matrix, cmap='binary', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()


def evaluate(particle):
    my_fitness = 0
    for i in range(len(particle)):
        # print(particle[i])
        for j in range(len(particle[i])):
            
            my_fitness += (i+j) * particle[i][j]
    #print(my_production)
    #my_production = 1000 - my_production
    return my_fitness

particles = np.random.uniform(low=0, high=1, size=(num_particles, plane_height, plane_width ))
particles = np.where(particles < 0.5, 0, 1)
gbest_score = np.inf
gbest = particles[0]


first_particle_fitness = []
average_finesses = []
first = True

for _ in range(iterations):
    fitnesses_in_iteration = []
    for i in range(num_particles):
        fitness = evaluate(particles[i])
        fitnesses_in_iteration.append(fitness)
        if fitness < gbest_score:
            gbest_score = fitness
            gbest = np.copy(particles[i])
        
    if first:
        first = False
    else:
        average_finesses.append(np.mean(fitnesses_in_iteration))
        first_particle_fitness.append(fitnesses_in_iteration[0])

    for i in range(num_particles):
        # Sebesség frissítése
        particles[i] = np.where(np.random.uniform(low=0, high=1, size=(plane_height, plane_width)) < 0.5, 0, 1)

    flat_indices = np.argsort(particles[i].ravel())[::-1]
    threshold_index = flat_indices[N]  # Get the index of the N-th largest value
    threshold_value = particles[i].ravel()[threshold_index]
    
    # Create a binary matrix based on the threshold value
    binary_matrix = np.where(particles[i] >= threshold_value, 1, 0)
    
    # In case of tie, randomly assign 1s to the tied values
    tied_indices = np.where(particles[i] == threshold_value)
    num_tied = len(tied_indices[0])
    if num_tied > N:
        chosen_indices = np.random.choice(num_tied, N, replace=False)
        binary_matrix[tied_indices] = 0
        binary_matrix[tied_indices[0][chosen_indices], tied_indices[1][chosen_indices]] = 1
    
    particles[i] = binary_matrix
    my_eval = evaluate(particles[i])


visualize_matrix(gbest, title="Global Best Particle")


plt.figure(figsize=(10, 5))
plt.plot(average_finesses, label='Average Fitness')
plt.xlabel('Iteration')
plt.ylabel('Average Fitness')
plt.title('Average Fitness Over Iterations')
plt.legend()
plt.show()