
# Project description

# Argument Parsing

from config import parse_arguments

parse_arguments()

from config import num
# Affinity

import os
import psutil
from config import thread_count
my_pid = os.getpid()
p = psutil.Process(my_pid)
starting_cpu = num*thread_count
cpu_list = [starting_cpu + i for i in range(thread_count)]
p.cpu_affinity(cpu_list)
print(p.cpu_affinity())

# Imports
import numpy as np
import time
import copy
import random
import math
import datetime
from matchingeq_functions import check_matching_equality
from Data.Classes.Data import Get_data
from Data.load_images import fire_input_paths, flood_input_paths, earthquake_input_paths, count_images
from Data.federated_learning import Servers_FL
from Data.Classes.Model import *
from Data.fl_parameters import *
from Classes.Server import Server
from Classes.User import User
from Classes.CriticalPoint import CP
from GT_Matching.utility_functions import user_utility_ext, server_utility_externality
from GT_Matching.approximate_matching import approximate_fedlearner_matching
from GT_Matching.accurate_matching import accurate_fedlearner_matching
from RL_Matching.rl_matching import rl_fedlearner_matching
from RAND_Matching.random_matching import random_fedlearner_matching

# General Parameters

from config import N,S,K
from general_parameters import *

federated_learning = True

# ===================== Users', Servers' and Critical Points' Topology ===================== #

start_time = time.time()

# Servers: on (0,0,0)
servers = []

while True:
    # Create servers with random p values
    for i in range(S):
        p = random.randint(math.ceil(0.333 * N), math.ceil(0.5 * N))
        server = Server(0,0,0,p,i)
        servers.append(server)

    # Ensure the sum of p is greater than or equal to N
    if sum(server.p for server in servers) >= N:
        break
    else:
        # If the condition is not met, empty the list and try again
        servers = []

for s in servers:
        print("Server", s.num, " Resources: ", s.p)

    
# Critical Points: inside a cube centered at (0,0,0) and side = 2
critical_points = []

i = 0

for i in range(K):

    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    z = random.uniform(-1,1)
    distance = math.sqrt(x**2 + y**2 + z**2)

    while distance < 0.5:  # While CP too close to the servers reselect
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        y = random.uniform(-1,1)
        distance = math.sqrt(x**2 + y**2 + z**2)

    disaster = disasters[i%S]
    
    cp = CP(x,y,z,i,disaster)

    critical_points.append(cp)

# Associate Critical Points with their Servers
for cp in critical_points:
    if(cp.get_disaster() == "fire"):
        servers[0].add_critical_point(cp)       # All fire CPs with Server 1
    elif(cp.get_disaster() == "flood"):
        servers[1].add_critical_point(cp)       # All flood CPs with Server 2
    else:
        servers[2].add_critical_point(cp)       # All earthquake CPs with Server 3

# Users: on a sphere (radius = 1) around the servers / users with lower i are closer to the CPs 
users = [] 

limit = 0.5

for i in range(N):

    while True:
        j = i/K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance > (j+1)*limit/N and distance <= (j+2)*limit/N:  # if in the desired sphere then add the user
            break
        
    user = User(x,y,z,i)
    
    users.append(user)

# ========================================================================================== #  
        
print()

# ===================== Initialization of Users' Utility Values ===================== #
        
# Data size for each user
        
Dn = [0]*N
        
for s in servers:   # For each server(disaster) calculate number of images each user will receive
    cps = s.get_critical_points()   # Get the relevant Critical Points
    user_min_distances = [-1]*N
    ratios = [0]*N

    image_num = 0

    total_images = count_images(fire_input_paths + flood_input_paths + earthquake_input_paths)

    # For each server count the images and select appropriate number of images to distribute
    if(s.num == 0): 
        image_num = count_images(fire_input_paths)
        ratio = image_num/total_images
        ratio = 1-math.sqrt(ratio)
        image_num = int(1.7*ratio*image_num)
        img_per_usr = image_num/N_max
    elif(s.num == 1):
        image_num = count_images(flood_input_paths)
        ratio = image_num/total_images
        ratio = 1-math.sqrt(ratio)
        image_num = int(1.3*ratio*image_num)
        img_per_usr = image_num/N_max
    else:
        image_num = count_images(earthquake_input_paths)
        ratio = image_num/total_images
        ratio = 1-math.sqrt(ratio)
        image_num = int(1.3*ratio*image_num)
        img_per_usr = image_num/N_max

    # For each user calculate the minimum distance from the relevant Critical Points
    for u in users:
      for cp in cps:
        user_x, user_y, user_z = u.x, u.y, u.z
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

        if(distance < user_min_distances[u.num] or user_min_distances[u.num] == -1):
            user_min_distances[u.num] = distance

    # Calculate the data size ratios based on the user minimum distance from the CPs
    for i in range(N):
        if user_min_distances[i] <= 1:
            ratios[i] = 1/(user_min_distances[i] + 1e-6)
        else:
            ratios[i] = 0

    ratios = [ratio/max(ratios) for ratio in ratios]

    # Get Sizes
    sizes = [int(1.8 * math.sqrt(ratio) * img_per_usr) for ratio in ratios]

    if sum(sizes) > image_num:
        temp_total = sum(sizes)
        sizes = [size*image_num/temp_total for size in sizes]

    print("Sizes:")
    print(sizes)

    # And add to the Dn of each user
    for i in range(N):
        Dn[i] += sizes[i]*3*224*224*8 # number of bits for all images

# Set the User final datasize
for i in range(N):
    Dn[i] += N_neutral*3*224*224*8  # each user has another 400 neutral images
    user = users[i]
    user.set_datasize(Dn[i])

print()
# =================================================================================== #
    
# Normalized Data Importance

min_dist = [None]*K
for i in range(K):
    for j in range(N):          # Finding Max Distance for each user
        user = users[j]
        user_x, user_y, user_z = user.x, user.y, user.z

        cp = critical_points[i]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

        if min_dist[i] is None or distance < min_dist[i]:
            min_dist[i] = distance

    for j in range(N):          # Calculating Importance for each user
        user = users[j]
        user_x, user_y, user_z = user.x, user.y, user.z

        cp = critical_points[i]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)   

        importance = min_dist[i] / distance

        user.add_importance(importance)
        
# ==================================================================  

# Normalized Data Rate

# Finding Max Data Rate
max_dr = 0
for i in range(N):
    for j in range(S):
        user = users[i]
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][i]

        g = 10**(-g / 10)       # path loss --> channel gain from db to power
        
        power = P[j][i] * distance
        
        dr = B*math.log2(1 + g*power/I0)
        
        if(dr > max_dr):
            max_dr = dr

# Calculating Normalized Data Rates
for i in range(N):
    for j in range(S):
        user = users[i]
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][i]

        g = 10**(-g / 10)       # path loss --> channel gain from db to power
        
        power = P[j][i] * distance
        
        dr = B*math.log2(1 + g*power/I0)
        
        user.add_datarate(dr/max_dr)


# Normalized Data Rate with Externality

# Finding Max Data Rate
max_dr = 0
for i in range(N):
    for j in range(S):
        user = users[i]
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][i]

        g = 10**(-g / 10)       # path loss --> channel gain from db to power
        
        power = P[j][i] * distance

        denominator_sum = 0
        for k in range(N):
            u = users[k]
            user_x, user_y, user_z = u.x, u.y, u.z
            distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
            g_ext = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][k]
            g_ext = 10**(-g_ext / 10)       # path loss --> channel gain from db to power
            denominator_sum += g_ext * P[j][k] * distance

        dr = B*math.log2(1 + g*power/(denominator_sum + I0))
        
        if(dr > max_dr):
            max_dr = dr

# Calculating Normalized Data Rates
for i in range(N):
    for j in range(S):
        user = users[i]
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][i]

        g = 10**(-g / 10)       # path loss --> channel gain from db to power

        power = P[j][i] * distance
        
        denominator_sum = 0
        for k in range(N):
            u = users[k]
            user_x, user_y, user_z = u.x, u.y, u.z
            distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
            g_ext = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][k]
            g_ext = 10**(-g_ext / 10)       # path loss --> channel gain from db to power
            denominator_sum += g_ext * P[j][k] * distance

        dr = B*math.log2(1 + g*power/(denominator_sum + I0))
        
        user.add_datarate_ext(dr/max_dr)
        
# ==================================================================

# Normalized User Payment

# Finding Max Payment
max_payment = 0
for i in range(S):
    server = servers[i]
    payment = server.p
    
    if(payment > max_payment):
        max_payment = payment
        
# Calculating Normalized Payments       
for i in range(N):
    user = users[i]
    for j in range(S):
        server = servers[j]
        payment = server.p
        
        user.add_payment(payment/max_payment)

# Normalize Payments for servers
for i in range(S):
    server = servers[i]
    server.set_p(server.p/max_payment)
        
# ==================================================================

# Normalized Energy Consumption

# Finding Max Local Energy Consumption
max_Elocal = 0
for i in range(N):
    user = users[i]
    E_local = user.get_Elocal()
    user.set_energy_ratio(E_local)
    if(E_local > max_Elocal):
        max_Elocal = E_local

# Calculating Normalized Local Energy Consumption
for i in range(N):
    user = users[i]
    E_local = user.get_Elocal()
    user.set_Elocal(E_local/max_Elocal)


# Normalized Energy Consumption to transmit the local model parameters to the server

# Find Max Transmission Energy
max_E_transmit = 0
for i in range(N):
    user = users[i]
    for j in range(S):
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][i]

        g = 10**(-g / 10)       # path loss --> channel gain from db to power
        
        power = P[j][i] * distance
        
        dr = B*math.log2(1 + g*power/I0)

        E_transmit = Z[i]*power/dr

        if(E_transmit > max_E_transmit):
            max_E_transmit = E_transmit

# Calculate Normalized Energy Transmission
for i in range(N):
    user = users[i]
    for j in range(S):
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][i]

        g = 10**(-g / 10)       # path loss --> channel gain from db to power
        
        power = P[j][i] * distance
        
        dr = B*math.log2(1 + g*power/I0)

        E_transmit = Z[i]*power/dr

        user.add_Etransmit(E_transmit/max_E_transmit)


# Normalized Energy Consumption to transmit the local model parameters to the server with Externality

# Find Max Transmission Energy
max_E_transmit = 0
for i in range(N):
    user = users[i]
    for j in range(S):
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][i]

        g = 10**(-g / 10)       # path loss --> channel gain from db to power
        
        power = P[j][i] * distance
        
        denominator_sum = 0
        for k in range(N):
            u = users[k]
            user_x, user_y, user_z = u.x, u.y, u.z
            distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
            g_ext = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][k]
            g_ext = 10**(-g_ext / 10)       # path loss --> channel gain from db to power
            denominator_sum += g_ext * P[j][k] * distance

        dr = B*math.log2(1 + g*power/(denominator_sum + I0))

        E_transmit = Z[i]*power/dr

        # Configure Energy Ratio for each user
        energy_ratio = user.get_energy_ratio()
        energy_ratio = energy_ratio/E_transmit
        user.set_energy_ratio(energy_ratio)

        if(E_transmit > max_E_transmit):
            max_E_transmit = E_transmit

# Calculate Normalized Energy Transmission
for i in range(N):
    user = users[i]
    for j in range(S):
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][i]

        g = 10**(-g / 10)       # path loss --> channel gain from db to power
        
        power = P[j][i] * distance
        
        denominator_sum = 0
        for k in range(N):
            u = users[k]
            user_x, user_y, user_z = u.x, u.y, u.z
            distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
            g_ext = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[j][k]
            g_ext = 10**(-g_ext / 10)       # path loss --> channel gain from db to power
            denominator_sum += g_ext * P[j][k] * distance

        dr = B*math.log2(1 + g*power/(denominator_sum + I0))

        E_transmit = Z[i]*power/dr

        user.add_Etransmit_ext(E_transmit/max_E_transmit)


# ==================================================================

# Normalized Data Quality

for i in range(K):

    # Finding Max Data Quality
    max_dq = 0
    for j in range(N):
        user = users[j]
        data_importance = user.get_importance()
        dq = data_importance[i] * user.get_datasize()
        if (dq > max_dq):
            max_dq = dq

    # Calculating Normalized Data Quality
    for j in range(N):
        user = users[j]
        data_importance = user.get_importance()
        dq = data_importance[i] * user.get_datasize()
        user.add_dataquality(dq/max_dq)        

# =================================================================================== #
        
for u in users:
    l=[]
    for s in servers:
        l.append(user_utility_ext(u,s,False))
    print(l) 
        
print()

ran_users = copy.deepcopy(users)
ran_servers = copy.deepcopy(servers)

# ================================= Random Matching ================================= #

ran_start = time.time()
    
random_fedlearner_matching(ran_users, ran_servers)

print("Random FedLearner Matching:\n")

for u in ran_users:
    allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
    print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

print()

ran_end = time.time()

print("Random Matching took", ran_end-ran_start, "seconds\n")
# =================================================================================== #

gt_users = copy.deepcopy(users)
gt_servers = copy.deepcopy(servers)

# ============================== Game Theory Matching ============================== #

# ============================== Approximate Matching ============================== #

gt_start = time.time()

# Initializing the available servers for each user
for i in range(N):
    u = gt_users[i]
    u.set_available_servers(gt_servers)
    
approximate_fedlearner_matching(gt_users, gt_servers)

print("Approximate FedLearner Matching:\n")

for u in gt_users:
    allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
    print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

print()

# ============================== Accurate Matching ============================== #
    
accurate_fedlearner_matching(gt_users, gt_servers)

print("Accurate FedLearner Matching:\n")

for u in gt_users:
    allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
    print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

print()

gt_end = time.time()

print("Game Theory Matching took", gt_end-gt_start, "seconds\n")
# =============================================================================== #

rl1_users = copy.deepcopy(users)
rl1_servers = copy.deepcopy(servers)

# =========================== RL Matching - Server Focused =========================== #
    
rl1_start = time.time()

rl_fedlearner_matching(rl1_users, rl1_servers, True)

print("Reinforcement Learning FedLearner Matching:\n")

for u in rl1_users:
    allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
    print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

print()

rl1_end = time.time()

print("Server Focused Reinforcement Learning Matching took", rl1_end-rl1_start, "seconds\n")
    
# ==================================================================================== #

rl2_users = copy.deepcopy(users)
rl2_servers = copy.deepcopy(servers)

# ========================== RL Matching - User Focused ========================== #

rl2_start = time.time()

rl_fedlearner_matching(rl2_users, rl2_servers, False)

print("Reinforcement Learning FedLearner Matching:\n")

for u in rl2_users:
    allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
    print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

print()

rl2_end = time.time()

print("User Focused Reinforcement Learning Matching took", rl2_end-rl2_start, "seconds\n")
    
# ================================================================================== #

# Prepare for the Trainings
user_lists = [ran_users, gt_users, rl1_users, rl2_users]
server_lists = [ran_servers, gt_servers, rl1_servers, rl2_servers]

matching_losses = []
matching_accuracies = []
matching_user_losses = []
matching_user_accuracies = []

# ============================== Federated Learning ============================== #

if federated_learning:
    prev_matchings = []

    get_data = Get_data(users, servers)

    X_train, y_train, X_server, y_server = get_data.pre_data()

    for _users, _servers in zip(user_lists, server_lists):

        same_matching = check_matching_equality(_servers, prev_matchings)

        elapsed_time = 0

        if same_matching is not None:
            server_losses, server_accuracy = matching_losses[same_matching], matching_accuracies[same_matching]
            user_losses, user_accuracy = matching_user_losses[same_matching], matching_user_accuracies[same_matching]
        else:
            X_train_copy = copy.deepcopy(X_train)
            y_train_copy = copy.deepcopy(y_train)
            X_test_copy = copy.deepcopy(X_server)
            y_test_copy = copy.deepcopy(y_server)
            learning_start = time.time()
            server_losses, server_accuracy, user_losses, user_accuracy = Servers_FL(_users, _servers, rounds, lr, epoch, X_train_copy, y_train_copy, X_test_copy, y_test_copy)
            learning_stop = time.time()
            elapsed_time = learning_stop - learning_start

        matching_losses.append(server_losses)
        matching_accuracies.append(server_accuracy)
        matching_user_losses.append(user_losses)
        matching_user_accuracies.append(user_accuracy)

        for i in range(S):
            print("Server ", i, " achieved:\n")
            print("Loss: ", server_losses[i][-1])
            print("Accuracy: ", server_accuracy[i][-1])
            print()
        print(f"Learning for all 3 Servers took {elapsed_time/60:.2f} minutes\n")
        print()

        prev_matchings.append(_servers)

    # ================================================================================ #
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nExecution took {elapsed_time/60:.2f} minutes\n")


# For each Matching log the metrics (Energy, Datarate, Utilities, Payments, Accuracy, Loss)

if federated_learning:
    # With Federated Learning
    matchings = []
    matchings.append((ran_users, ran_servers, matching_losses[0], matching_accuracies[0], matching_user_losses[0], matching_user_accuracies[0], "RAN"))
    matchings.append((gt_users, gt_servers, matching_losses[1], matching_accuracies[1], matching_user_losses[1], matching_user_accuracies[1], "GT"))
    matchings.append((rl1_users, rl1_servers, matching_losses[2], matching_accuracies[2], matching_user_losses[2], matching_user_accuracies[2], "RL1"))
    matchings.append((rl2_users, rl2_servers, matching_losses[3], matching_accuracies[3], matching_user_losses[3], matching_user_accuracies[3], "RL2"))
else:
    # Without Federated Learning
    matchings = []
    matchings.append((ran_users, ran_servers, "RAN"))
    matchings.append((gt_users, gt_servers, "GT"))
    matchings.append((rl1_users, rl1_servers, "RL1"))
    matchings.append((rl2_users, rl2_servers, "RL2"))

timestamp = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")

directory_path = "../results"       # Results Directory

# Check if the directory exists
if not os.path.exists(directory_path):
    # If it doesn't exist, create the directory
    os.makedirs(directory_path)

for matching in matchings:

    if federated_learning:
        _users, _servers, _losses, _accuracies, user_losses, user_accuracies, matching_label = matching     # With Federated Learning
    else:
        _users, _servers, matching_label = matching     # Without Federated Learning

    # Energy (J)
    mean_Energy = 0
    matched_users = 0
    for u in _users:
        if u.get_alligiance() is not None:
            matched_users += 1
            mean_Energy += u.get_Elocal() * max_Elocal
            mean_Energy += u.get_Etransmit_ext()[u.get_alligiance().num] * max_E_transmit

    mean_Energy /= matched_users

    # Local Energy (J)
    mean_Elocal = 0
    for u in _users:
        if u.get_alligiance() is not None:
            mean_Elocal += u.get_Elocal() * max_Elocal

    mean_Elocal /= matched_users

    # Transfer Energy (J)
    mean_Etransfer = 0
    for u in _users:
        if u.get_alligiance() is not None:
            mean_Etransfer += u.get_Etransmit_ext()[u.get_alligiance().num] * max_E_transmit

    mean_Etransfer /= matched_users

    # Datarate (bps)
    mean_Datarate = 0
    for u in _users:
        if u.get_alligiance() is not None:
            mean_Datarate += u.get_datarate_ext()[u.get_alligiance().num] * max_dr

    mean_Datarate /= matched_users

    # User Utility
    mean_User_Utility = 0
    for u in _users:
        if u.get_alligiance() is not None:
            mean_User_Utility += user_utility_ext(u, u.get_alligiance())

    mean_User_Utility /= matched_users

    # Server Utility
    mean_Server_Utility = 0
    for s in _servers:
        mean_Server_Utility += server_utility_externality(_servers, s.get_coalition(), s)

    mean_Server_Utility /= S

    # User Payments
    user_payments = 0
    for u in _users:
        if u.get_alligiance() is not None:
            user_payments += u.get_payment()[u.get_alligiance().num]*max_payment/len(list(u.get_alligiance().get_coalition()))

    output_filename = f"../results/u{N}_cp{K}_{timestamp}.txt"  # Choose a desired filename

    with open(output_filename, 'a') as file:
        file.write(f"Matching: {matching_label}, Users: {N}, Critical Points: {K}\n\
    Mean Energy: {mean_Energy} J\n\
    Mean Elocal: {mean_Elocal} J\n\
    Mean Etransfer: {mean_Etransfer} J\n\
    Mean Datarate: {mean_Datarate} bps\n\
    Mean User Utility: {mean_User_Utility}\n\
    Mean Server Utility: {mean_Server_Utility}\n\
    Sum User Payments: {user_payments}\n\
    \n")
        
    if federated_learning: 
        with open(output_filename, 'a') as file:
            file.write(f"Fire Server:\n\
        Losses: {_losses[0]}\n\
        Accuracies: {_accuracies[0]}\n")
            for u in _servers[0].get_coalition():
                file.write(f"User {u.num}:\n\
            Losses: {user_losses[u.num]}\n\
            Accuracies: {user_accuracies[u.num]}\n")
            
            file.write(f"Flood Server:\n\
        Losses: {_losses[1]}\n\
        Accuracies: {_accuracies[1]}\n")
            for u in _servers[1].get_coalition():
                file.write(f"User {u.num}:\n\
            Losses: {user_losses[u.num]}\n\
            Accuracies: {user_accuracies[u.num]}\n")
            
            file.write(f"Earthquake Server:\n\
        Losses: {_losses[2]}\n\
        Accuracies: {_accuracies[2]}\n")
            for u in _servers[2].get_coalition():
                file.write(f"User {u.num}:\n\
            Losses: {user_losses[u.num]}\n\
            Accuracies: {user_accuracies[u.num]}\n")