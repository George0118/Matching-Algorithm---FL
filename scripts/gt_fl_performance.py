import os
import re
import matplotlib.pyplot as plt


# Function to parse accuracies and losses from log files
def parse_log_file(file_path):
    accuracies = {}
    losses = {}
    matching = {}

    flag = False

    with open(file_path, 'r') as file:
        for line in file:
            if 'Matching:' in line:
                if 'GT'in line:
                    flag = True
                else:
                    flag = False

            if flag:
                if 'Fire Server' in line or 'Flood Server' in line or 'Earthquake Server' in line:
                    current_server = re.search(r'Fire Server|Flood Server|Earthquake Server', line).group()
                    losses[current_server] = [float(x) for x in re.findall(r'Losses: \[([\d.,\s]+)\]', next(file))[0].split(', ')][:40]
                    accuracies[current_server] = [float(x) for x in re.findall(r'Accuracies: \[([\d.,\s]+)\]', next(file))[0].split(', ')][:40]

                match_user = re.search(r'User \d+', line)
                if match_user:
                    user = match_user.group()
                    user_losses = [float(x) for x in re.findall(r'Losses: \[([\d.e,+,\-\s]+)\]', next(file))[0].split(', ')]
                    user_accuracies = [float(x) for x in re.findall(r'Accuracies: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                    accuracies[user] = user_accuracies[:40]
                    losses[user] = user_losses[:40]
                    matching[user] = current_server

    return accuracies, losses, matching

colors = {'Fire Server': 'r', 'Flood Server': 'b', 'Earthquake Server': 'g'}

# Function to plot data
def plot_data(data, title, save_dir=None):
    plt.figure(figsize=(10, 8))
    for key, value in data.items():
        if 'average_accuracies' in value:
            plt.plot(value['average_accuracies'], label=f'{key} (Average)', linewidth=2.5, color=colors[key])
        elif 'average_losses' in value:
            plt.plot(value['average_losses'], label=f'{key} (Average)', linewidth=2.5, color=colors[key])
        else:
            plt.plot(value, label=key, color=colors[key], linewidth=2.5)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy' if 'Accuracies' in title else 'Loss', fontsize=18)
    plt.xticks(fontsize=16) 
    plt.yticks(fontsize=16)
    if 'Accuracies' in title:
        plt.legend(fontsize=16)
    else:
        plt.legend(loc='upper left', fontsize = 16)
    plt.grid(True)
    
    # Save plot if save_dir is provided
    if save_dir:
        filename = title.replace(' ', '_') + '.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

# Directory containing log files
directory = '../../results/GT_FL_results'
save_directory = './gt_fl_performance'

# Initialize dictionaries to store accumulated data across all files
all_server_accuracies = {}
all_user_accuracies_per_server = {}
all_server_losses = {}
all_user_losses_per_server = {}

unique_servers = ['Fire Server', 'Flood Server', 'Earthquake Server']

# Iterate over files in directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        accuracies, losses, matching = parse_log_file(file_path)

        # Aggregate User accuracies and User losses for each server
        for user, server in matching.items():
            if server not in all_user_accuracies_per_server:
                all_user_accuracies_per_server[server] = {'accuracies': [], 'count': 0}
                all_user_losses_per_server[server] = {'losses': [], 'count': 0}

            all_user_accuracies_per_server[server]['accuracies'].append(accuracies[user])
            all_user_accuracies_per_server[server]['count'] += 1

            all_user_losses_per_server[server]['losses'].append(losses[user])
            all_user_losses_per_server[server]['count'] += 1

        # Aggregate Server accuracies and Server losses for each server
        for server in unique_servers:
            if server not in all_server_accuracies:
                all_server_accuracies[server] = {'accuracies': [], 'count': 0}
                all_server_losses[server] = {'losses': [], 'count': 0}

            all_server_accuracies[server]['accuracies'].append(accuracies[server])
            all_server_accuracies[server]['count'] += 1

            all_server_losses[server]['losses'].append(losses[server])
            all_server_losses[server]['count'] += 1  

# Calculate average user accuracies and losses per server across all files
average_all_user_accuracies_per_server = {}
average_all_user_losses_per_server = {}

for server, data in all_user_accuracies_per_server.items():
    average_all_user_accuracies_per_server[server] = [sum(item) / data['count'] for item in zip(*data['accuracies'])]

for server, data in all_user_losses_per_server.items():
    average_all_user_losses_per_server[server] = [sum(item) / data['count'] for item in zip(*data['losses'])]

average_all_server_accuracies = {}
average_all_server_losses = {}

for server, data in all_server_accuracies.items():
    average_all_server_accuracies[server] = [sum(item) / data['count'] for item in zip(*data['accuracies'])]

for server, data in all_server_losses.items():
    average_all_server_losses[server] = [sum(item) / data['count'] for item in zip(*data['losses'])]

# Plot aggregated and averaged data
plot_data(average_all_server_accuracies, 'Server Accuracies', save_directory)
plot_data(average_all_user_accuracies_per_server, 'User Accuracies', save_directory)
plot_data(average_all_server_losses, 'Server Losses', save_directory)
plot_data(average_all_user_losses_per_server, 'User Losses', save_directory)
