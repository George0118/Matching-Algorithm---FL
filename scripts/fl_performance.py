import os
import re
import matplotlib.pyplot as plt

results_directory = "../../results/accuracy_results"
output_directory = "./fl_performance/"
user_server_data = {}

# Function to extract a list of float values from a line
def extract_list(line):
    match = re.search(r'\[([^\]]+)\]', line)
    if match:
        values = match.group(1).split(', ')
        return [float(value) for value in values]
    return None

# Loop through files in the results directory
for filename in os.listdir(results_directory):
    file_path = os.path.join(results_directory, filename)

    # Check if the path is a file
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            # Read lines from the file
            lines = file.readlines()

            users = server_name = matching = losses = accuracies = None
            # Extract relevant data from each line
            for line in lines:
                users_match = re.match(r'Matching: (\w+), Users: (\d+),', line)
                if users_match:
                    matching = users_match.group(1)
                    users = users_match.group(2)
                server_match = re.match(r'\s*(\w+) Server:', line)
                if server_match:
                    server_name = server_match.group(1)
                elif "Losses:" in line:
                    losses = extract_list(line)
                elif "Accuracies:" in line:
                    accuracies = extract_list(line)

                # Check if all relevant data is available
                if users and server_name and matching and losses and accuracies:
                    key = f"Users: {users}, Server: {server_name}"
                    user_server_data.setdefault(key, {"Losses": {}, "Accuracies": {}})
                    user_server_data[key]["Losses"][matching] = losses
                    user_server_data[key]["Accuracies"][matching] = accuracies

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Create separate plots for Losses and Accuracies for each (Users, Server) combination
for key, data in user_server_data.items():
    plt.figure(figsize=(10, 6))

    # Plot Losses for each Matching type
    plt.subplot(2, 1, 1)
    plt.title(f"Losses for {key}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    epochs = list(range(1, len(next(iter(data["Losses"].values()))) + 1))
    all_losses = [loss for losses in data["Losses"].values() for loss in losses]
    plt.ylim(min(all_losses) - 0.01, max(all_losses) + 0.01)

    for matching, losses in data["Losses"].items():
        plt.plot(epochs, losses, label=f"{matching} Losses")

    plt.legend()

    # Plot Accuracies for each Matching type
    plt.subplot(2, 1, 2)
    plt.title(f"Accuracies for {key}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    all_accuracies = [accuracy for accuracies in data["Accuracies"].values() for accuracy in accuracies]
    plt.ylim(min(all_accuracies) - 0.01, max(all_accuracies) + 0.01)

    for matching, accuracies in data["Accuracies"].items():
        plt.plot(epochs, accuracies, label=f"{matching} Accuracies")

    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_filename = f"{key.replace(':', '_')}.png"  # Replace ':' in key to avoid issues in filename
    output_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_path)
    plt.close()  # Close the plot to free memory

print("Plots saved successfully.")
