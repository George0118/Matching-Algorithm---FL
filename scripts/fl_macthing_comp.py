import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to parse accuracies and losses from log files
def parse_log_file(file_path):
    matching = {}

    with open(file_path, 'r') as file:
        for line in file:
            if 'Matching:' in line:
                matching_type = line.split('Matching: ')[1].split(',')[0].strip()
                if matching_type not in matching:
                    matching[matching_type] = {}

            if 'Fire Server' in line or 'Flood Server' in line or 'Earthquake Server' in line:
                current_server = re.search(r'Fire Server|Flood Server|Earthquake Server', line).group()
                loss_values = [float(x) for x in re.findall(r'Losses: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                accuracy_values = [float(x) for x in re.findall(r'Accuracies: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                if current_server not in matching[matching_type]:
                    matching[matching_type][current_server]={'accuracies': [], 'losses': []}
                matching[matching_type][current_server]['losses'].append(loss_values[-1])  # Get the last loss value
                matching[matching_type][current_server]['accuracies'].append(accuracy_values[-1])  # Get the last accuracy value

    return matching

# Directory containing log files
directory = '../../results/FL_matching_comp'
save_directory = './fl_matching_comp'
os.makedirs(save_directory, exist_ok=True)

colors = {'Fire Server': 'r', 'Flood Server': 'b', 'Earthquake Server': 'g'}

# Iterate over files in directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        matching = parse_log_file(file_path)

# Plotting accuracies
matching_types = list(matching.keys())
server_types = list(matching[matching_types[0]].keys())
num_matching_types = len(matching_types)
num_server_types = len(server_types)
bar_width = 0.2
index = np.arange(num_matching_types)

# Plotting accuracies
plt.figure(figsize=(10, 5))
for i, server_type in enumerate(server_types):
    accuracies = [matching[matching_type][server_type]['accuracies'][-1] for matching_type in matching_types]
    plt.bar(index + i * bar_width, accuracies, bar_width, label=server_type, color=colors[server_type])

plt.xlabel('Matching Types')
plt.ylabel('Accuracy')
plt.title('Accuracy by Matching Types')
plt.xticks(index + bar_width * (num_server_types - 1) / 2, matching_types)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'accuracy_plot.png'))  # Save the plot
plt.show()

# Plotting losses
plt.figure(figsize=(10, 5))
for i, server_type in enumerate(server_types):
    losses = [matching[matching_type][server_type]['losses'][-1] for matching_type in matching_types]
    plt.bar(index + i * bar_width, losses, bar_width, label=server_type, color=colors[server_type])

plt.xlabel('Matching Types')
plt.ylabel('Loss')
plt.title('Loss by Matching Types')
plt.xticks(index + bar_width * (num_server_types - 1) / 2, matching_types)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'loss_plot.png'))  # Save the plot
plt.show()