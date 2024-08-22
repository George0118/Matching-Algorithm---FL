import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to parse accuracies and losses from log files
def parse_log_file(file_path):
    accuracies = {'URBAN': {}, 'SUBURBAN': {}, 'RURAL': {}}
    losses = {'URBAN': {}, 'SUBURBAN': {}, 'RURAL': {}}

    with open(file_path, 'r') as file:
        for line in file:
            if 'Matching:' in line:
                matching = line.split(":")[1].strip().split(",")[0]

            if 'Fire Server' in line or 'Flood Server' in line or 'Earthquake Server' in line:
                current_server = re.search(r'Fire Server|Flood Server|Earthquake Server', line).group()
                loss_values = [float(x) for x in re.findall(r'Losses: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                accuracy_values = [float(x) for x in re.findall(r'Accuracies: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                losses[matching][current_server] = loss_values[-1]  # Get the last loss value
                accuracies[matching][current_server] = accuracy_values[-1]  # Get the last accuracy value

    return accuracies, losses


# Directory containing log files
directory = '../../results/areas_results'
save_directory = './fl-areas/areas'

scalability_accuracies = {'URBAN': {}, 'SUBURBAN': {}, 'RURAL': {}}
scalability_losses = {'URBAN': {}, 'SUBURBAN': {}, 'RURAL': {}}

colors = {'Fire Server': 'r', 'Flood Server': 'b', 'Earthquake Server': 'g'}

# Iterate over files in directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        accuracies, losses = parse_log_file(file_path)

        pattern = r'([a-zA-Z]+)-u(\d+)_'
        match = re.search(pattern, filename)
        num_users = match.group(2)

        for matching, data in accuracies.items():
            for server, accuracy in data.items():
                if server not in scalability_accuracies[matching]:
                    scalability_accuracies[matching][server] = {'accuracies': [], 'users': []}
                    scalability_losses[matching][server] = {'losses': [], 'users': []}

                scalability_accuracies[matching][server]['accuracies'].append(data[server])
                scalability_accuracies[matching][server]['users'].append(num_users) 
                scalability_losses[matching][server]['losses'].append(losses[matching][server])
                scalability_losses[matching][server]['users'].append(num_users) 

# Initialize dictionaries to store cumulative totals and counts for accuracies and losses
average_accuracies = {'URBAN': {}, 'SUBURBAN': {}, 'RURAL': {}}
average_losses = {'URBAN': {}, 'SUBURBAN': {}, 'RURAL': {}}

# Function to calculate averages
def calculate_averages(data, averages, str):
    for area, area_data in data.items():
        for server, server_data in area_data.items():
            for value, num_users in zip(server_data[str], server_data['users']):
                if server not in averages[area]:
                    averages[area][server] = {}
                if num_users not in averages[area][server]:
                    averages[area][server][num_users] = {'total': 0, 'count': 0}
                averages[area][server][num_users]['total'] += value
                averages[area][server][num_users]['count'] += 1

# Calculate average accuracies
calculate_averages(scalability_accuracies, average_accuracies, 'accuracies')

# Calculate average losses
calculate_averages(scalability_losses, average_losses, 'losses')

# Calculate the average accuracies and losses for each num_users and server
for area, area_data in average_accuracies.items():
    for server, server_data in area_data.items():
        for num_users, accuracy_stats in server_data.items():
            # Calculate the average accuracy and loss
            average_accuracy = accuracy_stats['total'] / accuracy_stats['count']
            average_loss = average_losses[area][server][num_users]['total'] / average_losses[area][server][num_users]['count']
            # Store the averages in a new key in the averages dictionaries
            average_accuracies[area][server][num_users]['average_accuracy'] = average_accuracy
            average_losses[area][server][num_users]['average_loss'] = average_loss

users = ["10", "12", "14", "16", "18", "20", "22", "24", "26", "28", "30"]

# Summing average accuracy for each server
sum_accuracy = {'URBAN': [0] * len(users), 'SUBURBAN': [0] * len(users), 'RURAL': [0] * len(users)}
sum_loss = {'URBAN': [0] * len(users), 'SUBURBAN': [0] * len(users), 'RURAL': [0] * len(users)}

for area, area_data in average_accuracies.items():
    for server, user_data in area_data.items():
        for user_count, metrics in user_data.items():
            sum_accuracy[area][int((int(user_count)-10)/2)] += metrics['average_accuracy']

for area, area_data in average_losses.items():
    for server, user_data in area_data.items():
        for user_count, metrics in user_data.items():
            sum_loss[area][int((int(user_count)-10)/2)] += metrics['average_loss']

print(sum_accuracy)

# Plotting
plt.figure(figsize=(10, 6))
for area, values in sum_accuracy.items():
    plt.plot(users, values, label=area, linewidth=2.5)

plt.xlabel('Number of Users', fontsize = 18)
plt.ylabel('Sum of Server Accuracies', fontsize = 18)
plt.legend(fontsize = 16)
plt.xticks(fontsize=16) 
plt.yticks(fontsize=16)
plt.grid(True)
# Save plot if save_dir is provided
if save_directory:
    os.makedirs(save_directory, exist_ok=True)
    filename = 'Sum_of_Accuracies_vs_Users.png'
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
else:
    plt.show()

plt.figure(figsize=(10, 6))
for area, values in sum_loss.items():
    plt.plot(users, values, label=area, linewidth = 2.5)

plt.xlabel('Number of Users', fontsize = 18)
plt.ylabel('Sum of Server Losses', fontsize = 18)
plt.legend(fontsize = 16)
plt.xticks(fontsize=16) 
plt.yticks(fontsize=16)
plt.grid(True)
# Save plot if save_dir is provided
if save_directory:
    os.makedirs(save_directory, exist_ok=True)
    filename = 'Sum_of_Losses_vs_Users.png'
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
else:
    plt.show()

for area, data in sum_accuracy.items():
    print(sum(data))


plt.figure(figsize=(10, 6))
for i, (area, values) in enumerate(sum_loss.items()):
    plt.bar(area, sum(values))
    plt.text(i, sum(values) - (0.02 * sum(values)), f'{sum(values):.4f}', ha='center', va='top', fontsize=18)

plt.xlabel('Areas', fontsize = 18)
plt.ylabel('Sum of Server Losses', fontsize = 18)
plt.xticks(fontsize=16) 
plt.yticks(fontsize=16)
plt.grid(True)
# Save plot if save_dir is provided
if save_directory:
    os.makedirs(save_directory, exist_ok=True)
    filename = 'Sum_of_Losses_per_Area.png'
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
else:
    plt.show()

plt.figure(figsize=(10, 6))
for i, (area, values) in enumerate(sum_accuracy.items()):
    plt.bar(area, sum(values))
    plt.text(i, sum(values) - (0.02 * sum(values)), f'{sum(values):.4f}', ha='center', va='top', fontsize=18)

plt.xlabel('Areas', fontsize = 18)
plt.ylabel('Sum of Server Accuracies', fontsize = 18)
plt.xticks(fontsize=16) 
plt.yticks(fontsize=16)
plt.grid(True)
# Save plot if save_dir is provided
if save_directory:
    os.makedirs(save_directory, exist_ok=True)
    filename = 'Sum_of_Accuracies_per_Area.png'
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
else:
    plt.show()
