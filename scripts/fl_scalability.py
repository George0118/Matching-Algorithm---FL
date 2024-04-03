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

# Function to plot data for each server
def plot_averages(data, title, save_dir=None):
    plt.figure(figsize=(10, 6))
    for server, server_data in data.items():
        num_users = list(server_data.keys())
        average_accuracies = [user_data['average_accuracy'] for user_data in server_data.values() if 'average_accuracy' in user_data]
        average_losses = [user_data['average_loss'] for user_data in server_data.values() if 'average_loss' in user_data]

        if average_accuracies:
            plt.plot(num_users, average_accuracies, label=f"{server} (Avg Accuracy)", color=colors[server])
        if average_losses:
            plt.plot(num_users, average_losses, label=f"{server} (Avg Loss)", color=colors[server])
                
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

# Initialize dictionaries to store cumulative totals and counts for accuracies and losses
average_accuracies = {}
average_losses = {}

# Function to calculate averages
def calculate_averages(data, averages, str):
    for server, server_data in data.items():
        for value, num_users in zip(server_data[str], server_data['users']):
            if server not in averages:
                averages[server] = {}
            if num_users not in averages[server]:
                averages[server][num_users] = {'total': 0, 'count': 0}
            averages[server][num_users]['total'] += value
            averages[server][num_users]['count'] += 1

# Calculate average accuracies
calculate_averages(scalability_accuracies, average_accuracies, 'accuracies')

# Calculate average losses
calculate_averages(scalability_losses, average_losses, 'losses')

# Calculate the average accuracies and losses for each num_users and server
for server, server_data in average_accuracies.items():
    for num_users, accuracy_stats in server_data.items():
        if num_users in average_losses.get(server, {}):
            # Avoid division by zero
            if accuracy_stats['count'] != 0 and average_losses[server][num_users]['count'] != 0:
                # Calculate the average accuracy and loss
                average_accuracy = accuracy_stats['total'] / accuracy_stats['count']
                average_loss = average_losses[server][num_users]['total'] / average_losses[server][num_users]['count']
                # Store the averages in a new key in the averages dictionaries
                average_accuracies[server][num_users]['average_accuracy'] = average_accuracy
                average_losses[server][num_users]['average_loss'] = average_loss


print(average_accuracies)
print(average_losses)

plot_averages(average_accuracies, "Scalability of GT Accuracies", save_directory)
plot_averages(average_losses, "Scalability of GT Losses", save_directory)
