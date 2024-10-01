import numpy as np
import random
import matplotlib.pyplot as plt

# PSO Parameters
num_particles = 50
iterations = 500
plane_width = 24
plane_height = 20
N = 10
X = 100  # Number of iterations without improvement before exiting

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

    no_improvement_counter = 0  # Counter for tracking iterations without improvement

    for iter_num in range(iterations):
        improved = False  # Track if there was improvement in this iteration

        for i in range(num_particles):
            fitness = evaluate(particles[i])
            if fitness < pbest_scores[i]:
                pbest_scores[i] = fitness
                pbest[i] = particles[i]
            if fitness < gbest_score:
                gbest_score = fitness
                gbest = np.copy(particles[i])
                improved = True  # Improvement occurred

        # # Check if there was no improvement in this iteration
        # if not improved:
        #     no_improvement_counter += 1
        # else:
        #     no_improvement_counter = 0  # Reset the counter when there's an improvement

        # # Early exit condition
        # if no_improvement_counter >= X:
        #     print(f"Early exit at iteration {iter_num} due to no improvement for {X} iterations.")
        #     break

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

# Hyperparameter search function
def find_best_params():
    w_values = np.linspace(0.1, 2.0, 10)  # 10 different values for w
    c1_values = np.linspace(0.1, 2.0, 10) # 10 different values for c1
    c2_values = np.linspace(0.1, 2.0, 10) # 10 different values for c2

    best_w, best_c1, best_c2 = 0, 0, 0
    best_gbest_score = np.inf
    num_trials = 5  # Number of trials to average out the results

    # Iterate over all combinations of w, c1, and c2
    for w in w_values:
        for c1 in c1_values:
            for c2 in c2_values:
                trial_scores = []
                for trial in range(num_trials):
                    print(f"Evaluating w={w}, c1={c1}, c2={c2}, trial={trial+1}")
                    gbest_score = pso(w, c1, c2)
                    trial_scores.append(gbest_score)
                    print(f"Trial {trial+1} gbest_score={gbest_score}")

                # Calculate the average score and check for outliers
                avg_score = np.mean(trial_scores)
                std_dev = np.std(trial_scores)
                filtered_scores = [score for score in trial_scores if (avg_score - 2 * std_dev) <= score <= (avg_score + 2 * std_dev)]
                final_score = np.mean(filtered_scores) if filtered_scores else avg_score

                print(f"Final averaged score for w={w}, c1={c1}, c2={c2} is {final_score}")

                # # Visualize the scores
                # plt.figure(figsize=(10, 5))
                # plt.plot(trial_scores, marker='o', linestyle='-', color='b', label='Trial Scores')
                # plt.axhline(y=avg_score, color='r', linestyle='--', label='Average Score')
                # plt.axhline(y=avg_score + 2 * std_dev, color='g', linestyle='--', label='Upper Bound')
                # plt.axhline(y=avg_score - 2 * std_dev, color='g', linestyle='--', label='Lower Bound')
                # plt.title(f'Scores for w={w}, c1={c1}, c2={c2}')
                # plt.xlabel('Trial')
                # plt.ylabel('Score')
                # plt.legend()
                # plt.show()

                if final_score < best_gbest_score:
                    best_gbest_score = final_score
                    best_w, best_c1, best_c2 = w, c1, c2
    
    print(f"Best Parameters -> w: {best_w}, c1: {best_c1}, c2: {best_c2}")
    print(f"Best gbest_score: {best_gbest_score}")

# Run the search
find_best_params()
