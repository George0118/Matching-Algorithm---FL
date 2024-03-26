import os
import re
import matplotlib.pyplot as plt

# Function to parse accuracies and losses from log files
def parse_log_file(file_path):
    accuracies = {}
    losses = {}

    flag = False

    with open(file_path, 'r') as file:
        for line in file:
            if 'Matching:' in line:
                if 'GT' in line:
                    flag = True
                else:
                    flag = False

            if flag:
                if 'Fire Server' in line or 'Flood Server' in line or 'Earthquake Server' in line:
                    current_server = re.search(r'Fire Server|Flood Server|Earthquake Server', line).group()
                    loss_values = [float(x) for x in re.findall(r'Losses: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                    accuracy_values = [float(x) for x in re.findall(r'Accuracies: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                    losses[current_server] = loss_values[-1]  # Get the last loss value
                    accuracies[current_server] = accuracy_values[-1]  # Get the last accuracy value

    return accuracies, losses

# Directory containing log files
directory = '../../results/GT_FL_scalability_results'
save_directory = './gt_fl_scalability'

# Initialize dictionaries to store data
last_losses = {}
last_accuracies = {}

# Iterate over files in directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        accuracies, losses = parse_log_file(file_path)

        pattern = r'u(\d+)_'
        match = re.search(pattern, filename)
        num_users = match.group(1)

        last_accuracies[num_users] = accuracies
        last_losses[num_users] = losses

# Function to plot data for each server
def plot_data(data, title, save_dir=None):
    plt.figure(figsize=(10, 6))
    for server, values in data.items():
        if 'Accuracies' in title:
            plt.plot(values['users'], values['accuracies'], label=server, color=colors[server])
        else:
            plt.plot(values['users'], values['losses'], label=server, color=colors[server])
        plt.title(title)
        plt.xlabel('Number of Users')
        plt.ylabel('Accuracy' if 'Accuracies' in title else 'Loss')
        plt.legend()
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
directory = '../../results/GT_FL_scalability_results'
save_directory = './gt_fl_scalability'

scalability_accuracies = {}
scalability_losses = {}

colors = {'Fire Server': 'r', 'Flood Server': 'b', 'Earthquake Server': 'g'}

# Iterate over files in directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        accuracies, losses = parse_log_file(file_path)

        pattern = r'u(\d+)_'
        match = re.search(pattern, filename)
        num_users = match.group(1)

        for server, accuracy in accuracies.items():
            if server not in scalability_accuracies:
                scalability_accuracies[server] = {'accuracies': [], 'users': []}
                scalability_losses[server] = {'losses': [], 'users': []}

            scalability_accuracies[server]['accuracies'].append(accuracies[server])
            scalability_accuracies[server]['users'].append(num_users) 
            scalability_losses[server]['losses'].append(losses[server])
            scalability_losses[server]['users'].append(num_users) 

plot_data(scalability_accuracies, "Scalability of GT Accuracies", save_directory)
plot_data(scalability_losses, "Scalability of GT Losses", save_directory)
