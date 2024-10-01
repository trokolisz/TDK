import numpy as np
import random
import matplotlib.pyplot as plt

# PSO Parameters
num_particles = 50
iterations = 100
w = 1.0      # inertia weight
c1 = 2.5     # cognitive component
c2 = 0.5     # social component

# Plane dimensions
plane_width = 20
plane_height = 10

# Object shapes (width, height), cost, and production
objects = [
    {'shape': (2, 3), 'cost': 10, 'production': 150},
    {'shape': (3, 1), 'cost': 5, 'production': 30},
    {'shape': (3, 3), 'cost': 12, 'production': 100},
]

# Production goal
production_goal = 1500

# Efficiency matrix (values between 0 and 1)
efficiency_matrix = np.random.rand(plane_height, plane_width)

class ObjectPlacer:
    def __init__(self, plane_width, plane_height, objects):
        self.plane_width = plane_width
        self.plane_height = plane_height
        self.objects = objects

    def random_positions(self, num_objects):
        return [{'type': random.randint(0, len(self.objects)-1), 
                 'position': (random.randint(0, self.plane_width-1), random.randint(0, self.plane_height-1)), 
                 'rotation': random.randint(0, 1)} for _ in range(num_objects)]

class FitnessEvaluator:
    def __init__(self, plane_width, plane_height, objects, efficiency_matrix, production_goal):
        self.plane_width = plane_width
        self.plane_height = plane_height
        self.objects = objects
        self.efficiency_matrix = efficiency_matrix
        self.production_goal = production_goal

    def evaluate(self, positions):
        total_production = 0
        total_cost = 0
        plane = np.zeros((self.plane_height, self.plane_width))
        penalty = 0

        for pos in positions:
            obj = self.objects[pos['type']]
            x, y = pos['position']
            width, height = obj['shape'] if pos['rotation'] == 0 else obj['shape'][::-1]

            if x + width > self.plane_width or y + height > self.plane_height:
                penalty += 1000
            else:
                if np.any(plane[y:y+height, x:x+width] > 0):
                    penalty += 1000
                else:
                    plane[y:y+height, x:x+width] = 1
                    efficiency = np.mean(self.efficiency_matrix[y:y+height, x:x+width])
                    scaled_production = obj['production'] * efficiency
                    total_production += scaled_production
                    total_cost += obj['cost']

        if total_production > self.production_goal:
            penalty += (total_production - self.production_goal)**2

        if total_production < self.production_goal:
            penalty += 10000 + (self.production_goal - total_production)

        return total_cost + penalty

class VisualizeSolution:
    def __init__(self, plane_width, plane_height, objects, efficiency_matrix):
        self.plane_width = plane_width
        self.plane_height = plane_height
        self.objects = objects
        self.efficiency_matrix = efficiency_matrix

    def draw_solution(self, positions):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.plane_width)
        ax.set_ylim(0, self.plane_height)
        ax.set_aspect('equal')

        ax.imshow(self.efficiency_matrix, cmap='Blues', extent=[0, self.plane_width, 0, self.plane_height], origin='lower')

        for pos in positions:
            obj = self.objects[pos['type']]
            x, y = pos['position']
            width, height = obj['shape'] if pos['rotation'] == 0 else obj['shape'][::-1]

            rect = plt.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor='cyan', alpha=0.7)
            ax.add_patch(rect)
            ax.text(x + width/2, y + height/2, f'{pos["type"]}', color='black', ha='center', va='center')

        plt.grid(True)
        plt.show()

class Particle:
    def __init__(self, positions):
        self.positions = positions
        self.velocity = np.zeros(len(positions))
        self.best_position = positions
        self.best_fitness = float('inf')

    def update_velocity(self, best_global_position, c1, c2, w):
        num_objects = len(self.positions)
        for i in range(num_objects):
            r1 = random.random()
            r2 = random.random()
            cognitive = c1 * r1 * (self.best_position[i]['position'][0] - self.positions[i]['position'][0])
            social = c2 * r2 * (best_global_position[i]['position'][0] - self.positions[i]['position'][0])
            self.velocity[i] = w * self.velocity[i] + cognitive + social

    def update_position(self, plane_width, plane_height):
        num_objects = len(self.positions)
        for i in range(num_objects):
            self.positions[i]['position'] = (
                int(self.positions[i]['position'][0] + self.velocity[i]), 
                self.positions[i]['position'][1]
            )
            self.positions[i]['position'] = (
                max(0, min(plane_width - 1, self.positions[i]['position'][0])),
                max(0, min(plane_height - 1, self.positions[i]['position'][1]))
            )
class PSO:
    def __init__(self, num_particles, iterations, w, c1, c2, plane_width, plane_height, objects, efficiency_matrix, production_goal):
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.plane_width = plane_width
        self.plane_height = plane_height
        self.object_placer = ObjectPlacer(plane_width, plane_height, objects)
        self.fitness_evaluator = FitnessEvaluator(plane_width, plane_height, objects, efficiency_matrix, production_goal)
        self.visualizer = VisualizeSolution(plane_width, plane_height, objects, efficiency_matrix)
        self.particles = []
        self.best_global_position = None
        self.best_global_fitness = float('inf')

    def initialize_swarm(self):
        for _ in range(self.num_particles):
            num_objects = random.randint(5, 10)
            positions = self.object_placer.random_positions(num_objects)
            particle = Particle(positions)
            self.particles.append(particle)

    def adjust_global_position(self, num_objects):
        """Ensure the global best position has the same number of objects as a particle."""
        if self.best_global_position is None:
            return  # Skip if no global position is set yet

        if len(self.best_global_position) < num_objects:
            for _ in range(num_objects - len(self.best_global_position)):
                # Add random objects to match the particle size
                self.best_global_position.append({'type': random.randint(0, len(objects)-1), 
                                                  'position': (random.randint(0, self.plane_width-1), random.randint(0, self.plane_height-1)), 
                                                  'rotation': random.randint(0, 1)})
        elif len(self.best_global_position) > num_objects:
            # Truncate global position to match the particle size
            self.best_global_position = self.best_global_position[:num_objects]

    def run(self):
        self.initialize_swarm()

        for iteration in range(self.iterations):
            for particle in self.particles:
                fitness = self.fitness_evaluator.evaluate(particle.positions)

                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.positions[:]

                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = particle.positions[:]

            for particle in self.particles:
                # Ensure the global best position has the same number of objects as the current particle
                self.adjust_global_position(len(particle.positions))

                particle.update_velocity(self.best_global_position, self.c1, self.c2, self.w)
                particle.update_position(self.plane_width, self.plane_height)

            print(f"Iteration {iteration + 1}/{self.iterations}, Best Fitness: {self.best_global_fitness}")

        self.visualizer.draw_solution(self.best_global_position)

# Run PSO
pso = PSO(num_particles, iterations, w, c1, c2, plane_width, plane_height, objects, efficiency_matrix, production_goal)
pso.run()
