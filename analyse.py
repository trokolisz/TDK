import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read and parse the log file
log_file = "pso_results.log"  # change this to your actual file path

pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([\d., ]+), (\d+\.\d+), \[([0-9, ]+)\]"
data = []

with open(log_file, "r") as file:
    for line in file:
        match = re.match(pattern, line.strip())
        if match:
            inputs = [float(i) for i in match.group(2).split(',')]
            std_dev = float(match.group(3))
            values = [int(x) for x in match.group(4).split(',')]
            data.append(inputs + [std_dev, values])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Input1', 'Input2', 'Input3', 'Best Score', 'Avg Score', 'Std Dev', 'Values'])

# Correlation analysis to see how inputs affect the Best and Avg Score
correlation_matrix = df[['Input1', 'Input2', 'Input3', 'Best Score', 'Avg Score']].corr()

print("Correlation matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap Between Inputs and Scores')
plt.show()

# Find the best input combinations (max Best Score)
best_combination = df[df['Best Score'] == df['Best Score'].max()]
print("Best input combination:")
print(best_combination[['Input1', 'Input2', 'Input3', 'Best Score']])

# Plot the distribution of Best Scores vs. each input
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.scatter(df['Input1'], df['Best Score'], color='blue')
plt.title('Best Score vs Input1')
plt.xlabel('Input1')
plt.ylabel('Best Score')

plt.subplot(1, 3, 2)
plt.scatter(df['Input2'], df['Best Score'], color='green')
plt.title('Best Score vs Input2')
plt.xlabel('Input2')
plt.ylabel('Best Score')

plt.subplot(1, 3, 3)
plt.scatter(df['Input3'], df['Best Score'], color='red')
plt.title('Best Score vs Input3')
plt.xlabel('Input3')
plt.ylabel('Best Score')

plt.tight_layout()
plt.show()

# Check how inputs affect the average score
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.scatter(df['Input1'], df['Avg Score'], color='blue')
plt.title('Avg Score vs Input1')
plt.xlabel('Input1')
plt.ylabel('Avg Score')

plt.subplot(1, 3, 2)
plt.scatter(df['Input2'], df['Avg Score'], color='green')
plt.title('Avg Score vs Input2')
plt.xlabel('Input2')
plt.ylabel('Avg Score')

plt.subplot(1, 3, 3)
plt.scatter(df['Input3'], df['Avg Score'], color='red')
plt.title('Avg Score vs Input3')
plt.xlabel('Input3')
plt.ylabel('Avg Score')

plt.tight_layout()
plt.show()
