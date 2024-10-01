import numpy as np
import random
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    filename='pso_results.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# PSO Parameters
num_particles = 25
iterations = 250
plane_width = 8
plane_height = 10
N = 10

# Function to visualize binary matrix
def visualize_matrix(matrix, title="Binary Matrix"):
    plt.figure(figsize=(5, 3))
    plt.imshow(matrix, cmap='binary', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

# Fitness evaluation function
def evaluate(particle):
    rows, cols = np.indices(particle.shape)
    return np.sum((rows + cols) * particle)

# PSO Function to run the optimization for given w, c1, c2
def pso(w, c1, c2):
    particles = np.random.uniform(low=0, high=1, size=(num_particles, plane_height, plane_width))
    particles = np.where(particles < 0.5, 0, 1)

    velocities = np.zeros((num_particles, plane_height, plane_width))
    pbest = np.copy(particles)
    pbest_scores = np.array([np.inf] * num_particles)
    gbest_score = np.inf
    gbest = particles[0]

    for iter_num in range(iterations):
        for i in range(num_particles):
            fitness = evaluate(particles[i])
            if fitness < pbest_scores[i]:
                pbest_scores[i] = fitness
                pbest[i] = particles[i]
            if fitness < gbest_score:
                gbest_score = fitness
                gbest = np.copy(particles[i])

        # Update the particles
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])

            flat_indices = np.argsort(velocities[i].ravel())[::-1]
            threshold_index = flat_indices[N]
            threshold_value = velocities[i].ravel()[threshold_index]
            
            binary_matrix = np.where(velocities[i] >= threshold_value, 1, 0)
            tied_indices = np.where(velocities[i] == threshold_value)
            num_tied = len(tied_indices[0])
            if num_tied > N:
                chosen_indices = np.random.choice(num_tied, N, replace=False)
                binary_matrix[tied_indices] = 0
                binary_matrix[tied_indices[0][chosen_indices], tied_indices[1][chosen_indices]] = 1

            velocities[i] = binary_matrix
            particles[i] = particles[i].astype(np.float64) + velocities[i]

            flat_indices = np.argsort(particles[i].ravel())[::-1]
            threshold_index = flat_indices[N]
            threshold_value = particles[i].ravel()[threshold_index]
            binary_matrix = np.where(particles[i] >= threshold_value, 1, 0)
            tied_indices = np.where(particles[i] == threshold_value)
            num_tied = len(tied_indices[0])
            if num_tied > N:
                chosen_indices = np.random.choice(num_tied, N, replace=False)
                binary_matrix[tied_indices] = 0
                binary_matrix[tied_indices[0][chosen_indices], tied_indices[1][chosen_indices]] = 1

            particles[i] = binary_matrix

    return gbest_score

# Hyperparameter search function with parallel processing
def run_trial(w, c1, c2, trial):
    gbest_score = pso(w, c1, c2)
    return gbest_score

def find_best_params():
    w_values = np.linspace(0.1, 2.0, 7)
    c1_values = np.linspace(0.1, 2.0, 7)
    c2_values = np.linspace(0.1, 2.0, 7)

    best_w, best_c1, best_c2 = 0, 0, 0
    best_gbest_score = np.inf
    num_trials = 20

    max_workers = 20
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for w in w_values:
            for c1 in c1_values:
                for c2 in c2_values:
                    futures = [executor.submit(run_trial, w, c1, c2, trial) for trial in range(num_trials)]
                    trial_scores = [future.result() for future in futures]

                    avg_score = np.mean(trial_scores)
                    std_dev = np.std(trial_scores)
                    filtered_scores = [score for score in trial_scores if (avg_score - 2 * std_dev) <= score <= (avg_score + 2 * std_dev)]
                    final_score = np.mean(filtered_scores) if filtered_scores else avg_score

                    # Log the result
                    logging.info(f'{w}, {c1}, {c2}, {final_score}, {avg_score}, {std_dev}, {trial_scores}')

                    if final_score < best_gbest_score:
                        best_gbest_score = final_score
                        best_w, best_c1, best_c2 = w, c1, c2

# Run the search
find_best_params()
