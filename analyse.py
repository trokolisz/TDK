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

# Plot the distribution of Best Scores vs. each input
# Plot the distribution of Best Scores vs. each input with input-specific average lines
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.scatter(df['Input1'], df['Best Score'], color='blue')
input1_avg = df.groupby('Input1')['Best Score'].mean()
plt.plot(input1_avg.index, input1_avg.values, color='black', linestyle='--', label='Avg for Input1')
plt.title('Best Score vs Input1')
plt.xlabel('W')
plt.ylabel('Best Score')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(df['Input2'], df['Best Score'], color='green')
input2_avg = df.groupby('Input2')['Best Score'].mean()
plt.plot(input2_avg.index, input2_avg.values, color='black', linestyle='--', label='Avg for Input2')
plt.title('Best Score vs Input2')
plt.xlabel('C1')
plt.ylabel('Best Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(df['Input3'], df['Best Score'], color='red')
input3_avg = df.groupby('Input3')['Best Score'].mean()
plt.plot(input3_avg.index, input3_avg.values, color='black', linestyle='--', label='Avg for Input3')
plt.title('Best Score vs Input3')
plt.xlabel('C2')
plt.ylabel('Best Score')
plt.legend()

plt.tight_layout()
plt.show()

# Check how inputs affect the average score with input-specific average lines
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.scatter(df['Input1'], df['Avg Score'], color='blue')
input1_avg_score = df.groupby('Input1')['Avg Score'].mean()
plt.plot(input1_avg_score.index, input1_avg_score.values, color='black', linestyle='--', label='Avg for Input1')
plt.title('Avg Score vs Input1')
plt.xlabel('Input1')
plt.ylabel('Avg Score')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(df['Input2'], df['Avg Score'], color='green')
input2_avg_score = df.groupby('Input2')['Avg Score'].mean()
plt.plot(input2_avg_score.index, input2_avg_score.values, color='black', linestyle='--', label='Avg for Input2')
plt.title('Avg Score vs Input2')
plt.xlabel('Input2')
plt.ylabel('Avg Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(df['Input3'], df['Avg Score'], color='red')
input3_avg_score = df.groupby('Input3')['Avg Score'].mean()
plt.plot(input3_avg_score.index, input3_avg_score.values, color='black', linestyle='--', label='Avg for Input3')
plt.title('Avg Score vs Input3')
plt.xlabel('Input3')
plt.ylabel('Avg Score')
plt.legend()

plt.tight_layout()
plt.show()

