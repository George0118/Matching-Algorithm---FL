import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to parse accuracies and losses from log files
def parse_log_file(matching, file_path):

    with open(file_path, 'r') as file:
        for line in file:
            if 'Matching:' in line:
                matching_type = line.split('Matching: ')[1].split('_')[0].strip()
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


# Directory containing log files
directory = '../../results/regret_matching_FL'
save_directory = './fl_matching_comp/regret'
os.makedirs(save_directory, exist_ok=True)

colors = {'Fire Server': 'r', 'Flood Server': 'b', 'Earthquake Server': 'g'}

# Iterate over files in directory
matching = {}
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        parse_log_file(matching, file_path)

# Plotting accuracies
matching_types = list(matching.keys())
server_types = list(matching[matching_types[0]].keys())
num_matching_types = len(matching_types)
num_server_types = len(server_types)
bar_width = 0.3
index = np.arange(num_matching_types)

# Plotting accuracies
plt.figure(figsize=(10, 6))
for i, server_type in enumerate(server_types):
    accuracies = [matching[matching_type][server_type]['accuracies'] for matching_type in matching_types]
    average_accuracies = [np.mean(acc) for acc in accuracies]
    plt.bar(index + i * bar_width, average_accuracies, bar_width, label=server_type, color=colors[server_type])
    # Adding value annotations on top of bars
    for j, acc in enumerate(average_accuracies):
        plt.text(index[j] + i * bar_width, acc, f'{acc:.4f}', ha='center', va='bottom', fontsize = 10)

plt.xlabel('Matching Types', fontsize = 16)
plt.ylabel('Accuracy', fontsize = 16)
plt.xticks(index + bar_width * (num_server_types - 1) / 2, ["Game Theory", "Regret CI", "Regret II"], fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(loc='lower center', fontsize = 12)
plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'accuracy_plot.png'))  # Save the plot
# plt.show()

# Plotting losses
plt.figure(figsize=(10, 6))
for i, server_type in enumerate(server_types):
    losses = [matching[matching_type][server_type]['losses'] for matching_type in matching_types]
    average_losses = [np.mean(loss) for loss in losses]
    plt.bar(index + i * bar_width, average_losses, bar_width, label=server_type, color=colors[server_type])
    # Adding value annotations on top of bars
    for j, loss in enumerate(average_losses):
        plt.text(index[j] + i * bar_width, loss, f'{loss:.4f}', ha='center', va='bottom', fontsize = 10)

plt.xlabel('Matching Types', fontsize = 16)
plt.ylabel('Loss', fontsize = 16)
plt.xticks(index + bar_width * (num_server_types - 1) / 2, ["Game Theory", "Regret CI", "Regret II"], fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(fontsize = 12)
plt.tight_layout()
plt.savefig(os.path.join(save_directory, 'loss_plot.png'))  # Save the plot
# plt.show()
