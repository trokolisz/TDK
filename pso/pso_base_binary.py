import numpy as np
import random
import matplotlib.pyplot as plt

num_particles = 50
iterations = 1000

w = 1.15      # inertia weight
c1 = 0.33     # cognitive component
c2 = 1.36     #social component


plane_width = 24
plane_height = 20




N = 10


def visualize_matrix(matrix, title="Binary Matrix"):
    plt.figure(figsize=(5, 3))
    plt.imshow(matrix, cmap='binary', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()




particles = np.random.uniform(low=0, high=1, size=(num_particles, plane_height, plane_width ))
particles = np.where(particles < 0.5, 0, 1)

velocities = np.zeros((num_particles, plane_height, plane_width))

pbest = np.copy(particles)
pbest_scores = np.array([np.inf] * num_particles)
gbest_score = np.inf
gbest = particles[0]



def evaluate(particle):
    my_fitness = 0
    rows, cols = np.indices(particle.shape)
    my_fitness = np.sum((rows + cols) * particle)

    #print(my_production)
    #my_production = 1000 - my_production
    return my_fitness

first_particle_fitness = []
average_finesses = []
first = True
for _ in range(iterations):
    fitnesses_in_iteration = []
    for i in range(num_particles):
        
        fitness = evaluate(particles[i])
        fitnesses_in_iteration.append(fitness)
        if fitness < pbest_scores[i]:
            pbest_scores[i] = fitness
            pbest[i] = particles[i]
        if fitness < gbest_score:
            gbest_score = fitness
            
            
            #visualize_matrix(particles[i], title=f"particlesNew gbest: {gbest_score}")
            gbest = np.copy(particles[i])
            #visualize_matrix(gbest, title=f"gbestNew gbest: {gbest_score}")
    #print(np.mean(fitnesses_in_iteration))
    if first:
        first = False
    else:
        average_finesses.append(np.mean(fitnesses_in_iteration))
        first_particle_fitness.append(fitnesses_in_iteration[0])
    #print(f"g score: {gbest_score}")
    #print(f"g ev: {evaluate(gbest)}")

    for i in range(num_particles):
        # Sebesség frissítése

        r1, r2 = np.random.rand(2)  # Generate r1, r2 once
        velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])

        #visualize_matrix(velocities[i], title=f"Velocity of particle {i} at iteration {_}")
        flat_indices = np.argsort(velocities[i].ravel())[::-1]
        threshold_index = flat_indices[N]  # Get the index of the N-th largest value
        threshold_value = velocities[i].ravel()[threshold_index]
        
        # Create a binary matrix based on the threshold value
        binary_matrix = np.where(velocities[i] >= threshold_value, 1, 0)
        
        # In case of tie, randomly assign 1s to the tied values
        tied_indices = np.where(velocities[i] == threshold_value)
        num_tied = len(tied_indices[0])
        if num_tied > N:
            chosen_indices = np.random.choice(num_tied, N, replace=False)
            binary_matrix[tied_indices] = 0
            binary_matrix[tied_indices[0][chosen_indices], tied_indices[1][chosen_indices]] = 1
        
        velocities[i] = binary_matrix
        #visualize_matrix(velocities[i], title=f"Velocity of particle {i} at iteration {_}")


        particles[i] = particles[i].astype(np.float64) + velocities[i]
        # visualize_matrix(particles[i], title=f"Particle {i} at iteration {_}")
        # Convert velocities[i] to a binary matrix


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
        #print(my_eval)
        #visualize_matrix(particles[i], title=f"Particle {i} at iteration {_}")



    #visualize_matrix(gbest, title=f"Best particle at iteration {_}")
    


print("Optimális pozíció (termelési mennyiségek):", gbest)
print("Maximális bevétel:", gbest_score)


# Visualize the global best particle
visualize_matrix(gbest, title="Global Best Particle")


plt.figure(figsize=(10, 5))
plt.plot(average_finesses, label='Average Fitness')
plt.xlabel('Iteration')
plt.ylabel('Average Fitness')
plt.title('Average Fitness Over Iterations')
plt.legend()
plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(first_particle_fitness, label='first Fitness')
# plt.xlabel('Iteration')
# plt.ylabel('first Fitness')
# plt.title('first Fitness Over Iterations')
# plt.legend()
# plt.show()