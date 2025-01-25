import os
import re
import matplotlib.pyplot as plt
import numpy as np

results_directory = "../../results/main"
matching_data = {}

# Function to extract a single float value from a line
def extract_value(line):
    match = re.search(r'\b\d+\.\d+\b', line)
    return float(match.group()) if match else None

# Loop through files in the results directory
for filename in os.listdir(results_directory):
    file_path = os.path.join(results_directory, filename)

    # Check if the path is a file
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            # Read lines from the file
            lines = file.readlines()

            matching = users = None
            # Extract relevant data from each line
            for line in lines:
                match = re.match(r'Matching: (\w+), Users: (\d+), Critical Points: (\d+)', line)
                if match:
                    matching, users = match.groups()[0], int(match.groups()[1])  # Store Matching and Users
                    matching_data.setdefault(matching, {"Data": {}})
                elif "Mean Energy" in line or "Mean Elocal" in line or "Mean Etransfer" in line \
                        or "Mean Datarate" in line or "Mean User Utility" in line or "Mean Server Utility" in line \
                        or "Sum User Payments" or "Time" in line:
                    magnitude_match = re.search(r'^\s*([^:]+)\s*:', line)
                    if magnitude_match:
                        magnitude = magnitude_match.group(1).strip()
                        value = extract_value(line)
                        matching_data[matching]["Data"].setdefault(magnitude, {"Users": [], "Values": []})
                        matching_data[matching]["Data"][magnitude]["Users"].append(users)  # Users
                        matching_data[matching]["Data"][magnitude]["Values"].append(value)

# Create line plots for each magnitude
magnitudes = ["Mean Energy", "Mean Elocal", "Mean Etransfer", "Mean Datarate", "Mean User Utility", "Mean Server Utility", "Mean Dataquality", "Time"]

plot_data = {}
for matching in matching_data.keys():
    plot_data[matching] = {}
    for magnitude in magnitudes:
        plot_data[matching][magnitude] = {"Users": [], "Values": []}
        

# Aggregate and average data for the same number of users across multiple runs
for matching, data in matching_data.items():
    for magnitude, values_data in data["Data"].items():
        users_values = {}
        for i, users in enumerate(values_data["Users"]):
            if users not in users_values:
                users_values[users] = []
            users_values[users].append(values_data["Values"][i])
        
        for users, values_list in users_values.items():
            averaged_value = np.mean(values_list)
            plot_data[matching][magnitude]["Users"].append(users)
            plot_data[matching][magnitude]["Values"].append(averaged_value)

user_values = [12, 15, 18, 21, 24, 27, 30]
save_directory = "./matching_comparison/main"
os.makedirs(save_directory, exist_ok=True)

for magnitude in magnitudes:
    plt.figure(figsize=(10, 6))
    plt.title(f"{magnitude} vs Users for Each Matching")
    plt.xlabel("Users")
    plt.ylabel(magnitude)

    for matching, data in plot_data.items():

        users = data[magnitude]["Users"]
        values = data[magnitude]["Values"]

        if matching == "RAN":
            m = "Random"
        elif matching == "GT":
            m = "Game Theory"
        elif matching == "RL1":
            m = "Server Focused RL"
        else:
            m = "User Focused RL"

        plt.plot(users, values, marker='o', label=f"Matching: {m}")

    plt.xticks(user_values)  # Set x ticks to predefined user values
    plt.legend()
    
    # Save the plot as PNG
    plt.savefig(os.path.join(save_directory, f"{magnitude.replace(' ', '_')}_vs_Users.png"), bbox_inches='tight')
    # plt.show()

# Create bar plots for average magnitude values across all matchings
for magnitude in magnitudes:
    plt.figure(figsize=(10, 6))
    plt.title(f"Average {magnitude} for Each Matching")
    plt.xlabel("Matching")
    plt.ylabel("Average " + magnitude)

    avg_values = []  # List to store average values for each matching
    bar_colors = ['blue', 'orange', 'green', 'red']  # Colors for the bars

    for matching, data in plot_data.items():
        values = data[magnitude]["Values"]
        avg_value = np.mean(values)
        avg_values.append(avg_value)

    # Plotting the bar plot with thinner bars
    bars = plt.bar(["Random", "Game Theory", "Server Focused RL", "User Focused RL"], avg_values, color=bar_colors, width=0.3)

    # Adding text labels on top of each bar
    for bar, value in zip(bars, avg_values):
        if magnitude == "Mean Etransfer":
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01*bar.get_height(), f'{value:.7f}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01*bar.get_height(), f'{value:.2f}', ha='center', va='bottom')

    # Save the plot as PNG
    plt.savefig(os.path.join(save_directory, f"Average_{magnitude.replace(' ', '_')}.png"), bbox_inches='tight')
    # plt.show()

