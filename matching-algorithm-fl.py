
# Project description

# Imports
from Classes.User import User
from Classes.Server import Server, importance_map
from Classes.CriticalPoint import CP
from GT_Matching.approximate_matching import approximate_fedlearner_matching
from GT_Matching.accurate_matching import accurate_fedlearner_matching
from RL_Matching.rl_matching import rl_fedlearner_matching
from Data.federated_learning import Servers_FL
from Data.load_images import fire_input_paths, flood_input_paths, earthquake_input_paths, count_images, factor
import numpy as np
import time
from Data.Classes.Model import *

# General Parameters

from general_parameters import *

# ===================== Users', Servers' and Critical Points' Topology ===================== #

import random
import math

start_time = time.time()

# Servers: on (0,0,0)
servers = []

for i in range(S):
    p = random.randint(int(0.333*N), int(0.5*N))
    server = Server(0,0,0,p,i)
    servers.append(server)
    
# Users: on a sphere (radius = 1) around the servers
users = [] 

for i in range(N):
    theta = random.uniform(0, 2 * math.pi)
    phi = math.acos(2 * random.uniform(0, 1) - 1)

    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    
    user = User(x,y,z,i)
    
    users.append(user)
    
# Critical Points: inside a cube centered at (0,0,0) and side = 2
critical_points = []

i = 0

while(len(critical_points) < K):
    flag = True  # Flag to check whether there is enough spacing for each disaster

    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    y = random.uniform(-1,1)
    disaster = disasters[i%S]
    
    cp = CP(x,y,z,i,disaster)

    for existing_cp in critical_points:
        if(existing_cp.num%S != i%S):
            existing_cp_x, existing_cp_y, existing_cp_z = existing_cp.x, existing_cp.y, existing_cp.z

            distance = math.sqrt((existing_cp_x - x)**2 + (existing_cp_y - y)**2 + (existing_cp_z - z)**2)
            if(distance < 1):
                flag = False
                break
        
    if flag:
        critical_points.append(cp)
        i += 1

# Associate Critical Points with their Servers
for cp in critical_points:
    if(cp.get_disaster() == "fire"):
        servers[0].add_critical_point(cp)       # All fire CPs with Server 1
    elif(cp.get_disaster() == "flood"):
        servers[1].add_critical_point(cp)       # All flood CPs with Server 2
    else:
        servers[2].add_critical_point(cp)       # All earthquake CPs with Server 3

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
        image_num = int(factor*ratio*image_num)
        print("Fire Images: ", image_num)
    elif(s.num == 1):
        image_num = count_images(flood_input_paths)
        ratio = image_num/total_images
        ratio = 1-math.sqrt(ratio)
        image_num = int(factor*ratio*image_num)
        print("Flood Images: ", image_num)
    else:
        image_num = count_images(earthquake_input_paths)
        ratio = image_num/total_images
        ratio = 1-math.sqrt(ratio)
        image_num = int(factor*ratio*image_num)
        print("Earthquake Images: ", image_num)

    # For each user calculate the minimum distance from the relevant Critical Points
    for u in users:
      for cp in cps:
        user_x, user_y, user_z = u.x, u.y, u.z
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

        if(distance < user_min_distances[u.num] or user_min_distances[u.num] == -1):
            user_min_distances[u.num] = distance

    print(user_min_distances)

    # Calculate the data size ratios based on the user minimum distance from the CPs
    for i in range(N):
      ratios[i] = 1/(user_min_distances[i] + 1e-6)

    print(ratios)

    sum_ratios = sum(ratios)

    # Get Sizes
    sizes = [int(ratio * image_num/sum_ratios) for ratio in ratios]

    print("Sizes:")
    print(sizes)

    # And add to the Dn of each user
    for i in range(N):
        Dn[i] += sizes[i]*3*224*224*8 # number of bits for all images

# Set the User final datasize
for i in range(N):
    user = users[i]
    user.set_datasize(Dn[i])

print()
# =================================================================================== #
    
# Normalized Data Importance

max_dist = [0]*K
for i in range(K):
    for j in range(N):          # Finding Max Distance for each user
        user = users[j]
        user_x, user_y, user_z = user.x, user.y, user.z

        cp = critical_points[i]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

        if distance > max_dist[i]:
            max_dist[i] = distance

    for j in range(N):          # Calculating Importance for each user
        user = users[j]
        user_x, user_y, user_z = user.x, user.y, user.z

        cp = critical_points[i]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)   

        importance = distance / max_dist[i]

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
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random.uniform(-1, 1)
        
        power = P[j][i] * distance
        
        dr = math.log2(1 + g*power/I0)
        
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
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random.uniform(-1, 1)
        
        power = P[j][i] * distance
        
        dr = math.log2(1 + g*power/I0)
        
        user.add_datarate(dr/max_dr)
        
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
        
# ==================================================================

# Normalized Energy Consumption

# Finding Max Local Energy Consumption
max_Elocal = 0
for i in range(N):
    user = users[i]
    E_local = user.get_Elocal()
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
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random.uniform(-1, 1)
        
        power = P[j][i] * distance
        
        dr = math.log2(1 + g*power/I0)

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
        
        g = 128.1 + 37.6 * np.log10(distance) + 8 * random.uniform(-1, 1)
        
        power = P[j][i] * distance
        
        dr = math.log2(1 + g*power/I0)

        E_transmit = Z[i]*power/dr

        user.add_Etransmit(E_transmit/max_E_transmit)


# ==================================================================

# Normalized Data Quality

for i in range(K):

    # Finding Max Data Quality
    max_dq = 0
    for j in range(N):
        user = users[j]
        data_importance = user.get_importance()
        dq = importance_map(data_importance[i]) * user.get_datasize()
        if (dq > max_dq):
            max_dq = dq

    # Calculating Normalized Data Quality
    for j in range(N):
        user = users[j]
        data_importance = user.get_importance()
        dq = importance_map(data_importance[i]) * user.get_datasize()
        user.add_dataquality(dq/max_dq)        

# =================================================================================== #

gt_users = users.copy()
gt_servers = servers.copy()

# ============================== Approximate Matching ============================== #

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
# =================================================================================== #


# ============================== Accurate Matching ============================== #
    
accurate_fedlearner_matching(gt_users, gt_servers)

print("Accurate FedLearner Matching:\n")

for u in gt_users:
    allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
    print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

print()
# =============================================================================== #

# ============================== Federated Learning ============================== #

rounds=100 # number of global rounds
lr=10e-4 # learning rate
epoch=1 # local iterations

server_losses, server_accuracy = Servers_FL(gt_users, gt_servers, rounds, lr, epoch)

for i in range(S):
    print("Server ", i, " achieved:\n")
    print("Loss: ", server_losses[i][-1])
    print("Accuracy: ", server_accuracy[i][-1])
    print()

# ================================================================================ #

rl_users = user.copy()  
rl_servers = servers.copy()  

# ============================== Reinforcment Learning ============================== #
    
rl_fedlearner_matching(rl_users, rl_servers)
    
# =================================================================================== #
    
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\nLearning took {elapsed_time/60:.2f} minutes\n")
