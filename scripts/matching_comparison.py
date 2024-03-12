import os
import re
import matplotlib.pyplot as plt

results_directory = "../../results"
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
                    matching_data.setdefault(matching, {"Users": users, "Data": {}})
                elif "Mean Energy" in line or "Mean Elocal" in line or "Mean Etransfer" in line \
                        or "Mean Datarate" in line or "Mean User Utility" in line or "Mean Server Utility" in line \
                        or "Sum User Payments" in line:
                    magnitude_match = re.search(r'^\s*([^:]+)\s*:', line)
                    if magnitude_match:
                        magnitude = magnitude_match.group(1).strip()
                        value = extract_value(line)
                        matching_data[matching]["Data"].setdefault(magnitude, {"Users": [], "Values": []})
                        matching_data[matching]["Data"][magnitude]["Users"].append(users)  # Users
                        matching_data[matching]["Data"][magnitude]["Values"].append(value)

# Create line plots for each magnitude
magnitudes = ["Mean Energy", "Mean Elocal", "Mean Etransfer", "Mean Datarate", "Mean User Utility", "Mean Server Utility", "Sum User Payments"]

for magnitude in magnitudes:
    plt.figure(figsize=(10, 6))
    plt.title(f"{magnitude} vs Users for Each Matching")
    plt.xlabel("Users")
    plt.ylabel(magnitude)

    for matching, data in matching_data.items():
        if magnitude in data["Data"]:
            users = data["Data"][magnitude]["Users"]
            values = data["Data"][magnitude]["Values"]

            plt.plot(users, values, marker='o', label=f"Matching: {matching}")

    plt.legend()
    plt.show()
