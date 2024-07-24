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
from helping_functions import dataset_sizes, h
from Data.Classes.Data import Get_data
from Data.load_images import fire_input_paths, flood_input_paths, earthquake_input_paths, count_images
from Data.federated_learning import Servers_FL
from Data.Classes.Model import *
from Data.fl_parameters import *
from Classes.Server import Server
from Classes.User import User, Ptrans_max, fn_max, E_local_max
from Classes.CriticalPoint import CP
from Regret_Matching.regret_matching import regret_matching
from Regret_Matching.regret_matching_II import regret_matching_II
from GT_Matching.approximate_matching import approximate_fedlearner_matching
from GT_Matching.accurate_matching import accurate_fedlearner_matching

# General Parameters

from config import N,S,K
from general_parameters import *

urban_threshold = 30    # In an Urban Area, after 30 users the users start to be further in proximity to the CPs
suburban_threshold = 21 # In an Suburban Area, after 21 users the users start to be further in proximity to the CPs
rural_threshold = 12    # In an Rural Area, after 12 users the users start to be further in proximity to the CPs
federated_learning = True

# ===================== Users', Servers' and Critical Points' Topology ===================== #

start_time = time.time()

# Servers: on (0,0,0)
servers = []

while True:
    # Create servers with random p values
    for i in range(S):
        p = math.ceil(0.333 * N)
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

for i in range(K):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = random.uniform(-1, 1)
    distance = math.sqrt(x**2 + y**2 + z**2)

    # Ensure that the distance between two critical points is at least 0.8
    while any(math.sqrt((x - cp.x)**2 + (y - cp.y)**2 + (z - cp.z)**2) < 0.8 for cp in critical_points) or distance < 0.3:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        distance = math.sqrt(x**2 + y**2 + z**2)

    disaster = disasters[i % S]

    cp = CP(x, y, z, i, disaster)
    critical_points.append(cp)

# Associate Critical Points with their Servers
for cp in critical_points:
    if(cp.get_disaster() == "fire"):
        servers[0].add_critical_point(cp)       # All fire CPs with Server 1
    elif(cp.get_disaster() == "flood"):
        servers[1].add_critical_point(cp)       # All flood CPs with Server 2
    else:
        servers[2].add_critical_point(cp)       # All earthquake CPs with Server 3

urban_servers = copy.deepcopy(servers)
subruban_servers = copy.deepcopy(servers)
rural_servers = copy.deepcopy(servers)

# Users: on a sphere (radius = 1) around the servers / users with lower i are closer to the CPs 
urban_users = [] 
suburban_users = [] 
rural_users = [] 

distance_diff = 0.005

for i in range(min(rural_threshold, N)):

    while True:
        j = i//K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.02+(j+1)*distance_diff and distance >= 0.02+j*distance_diff:  # if in the desired sphere and good user then add the user
            break

    user = User(x,y,z,i)
    
    urban_users.append(copy.deepcopy(user))
    suburban_users.append(copy.deepcopy(user))
    rural_users.append(copy.deepcopy(user))

for i in range(min(suburban_threshold - rural_threshold, N - rural_threshold)):

    while True:
        j = i/K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.02+(rural_threshold+j+1)*distance_diff and distance >= 0.02+(rural_threshold+j)*distance_diff:  # if in the desired sphere and good user then add the user
            break
        
    user = User(x,y,z,i+rural_threshold)
    
    urban_users.append(copy.deepcopy(user))
    suburban_users.append(copy.deepcopy(user))

    while True:
        j = i/K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.3+(j+1)*distance_diff and distance >= 0.3+j*distance_diff:     # if in the desired sphere and bad user then add the user
            break
        
    user = User(x,y,z,i+rural_threshold)
    
    rural_users.append(copy.deepcopy(user))


for i in range(min(rural_threshold - suburban_threshold, N - suburban_threshold)):

    while True:
        j = i/K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.02+(suburban_threshold+j+1)*distance_diff and distance >= 0.02+(suburban_threshold+j)*distance_diff:  # if in the desired sphere and good user then add the user
            break
        
    user = User(x,y,z,i+suburban_threshold)
    
    urban_users.append(copy.deepcopy(user))

    while True:
        j = i/K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.2+(j+1)*distance_diff and distance >= 0.2+j*distance_diff:     # if in the desired sphere and bad user then add the user
            break
        
    user = User(x,y,z,i+suburban_threshold)
    
    suburban_users.append(copy.deepcopy(user))

    while True:
        j = i/K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.3+(suburban_threshold-rural_threshold+j+1)*distance_diff and distance >= 0.3+(suburban_threshold-rural_threshold+j)*distance_diff:     # if in the desired sphere and bad user then add the user
            break
        
    user = User(x,y,z,i+suburban_threshold)
    
    rural_users.append(copy.deepcopy(user))

# ========================================================================================== #  
        
all_users = {'Urban': urban_users, 'Suburban': suburban_users, 'Rural': rural_users}
print()

# ===================== Initialization of Users' Utility Values ===================== #

# Set Max Payments

for area, users in all_users.items(): 
    for i in range(N):
        user = users[i]
        for j in range(S):
            user.max_payment.append(servers[j].p)
        
# Data size for each user
        


total_images = count_images(fire_input_paths + flood_input_paths + earthquake_input_paths)

# For each user calculate the minimum distance from the relevant Critical Points
for area, users in all_users.items():
    Dn = [0]*N
    
    for s in servers:   # For each server(disaster) calculate number of images each user will receive
        cps = s.get_critical_points()   # Get the relevant Critical Points
        sizes = dataset_sizes(s, area, users, cps, total_images)

        # And add to the Dn of each user
        for i in range(N):
            Dn[i] += sizes[i]*3*224*224*8 # number of bits for all images

    # Set the User final datasize
    for i in range(N):
        Dn[i] += N_neutral*3*224*224*8  # each user has another 250 neutral images
        user = users[i]
        user.set_datasize(Dn[i])

print()

# =================================================== #

# Set distances

for area, users in all_users.items():
    for i in range(N):
        distances = []
        for j in range(S):          # Finding Max Distance for each user
            user = users[i]
            user_x, user_y, user_z = user.x, user.y, user.z

            s = servers[j]
            server_x, server_y, server_z = s.x, s.y, s.z

            distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)

            distances.append(distance)
        user.set_distances(distances)

# ===================================================== #
# =================================================================================== #
    
# Normalized Data Importance

for area, users in all_users.items():
    min_dist = [None]*K
    for i in range(K):
        for j in range(N):          # Finding Min Distance for each CP
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

        print(user.get_importance())

# ======================================================================================== #

# ========== Calculating parameters for utility functions =========== #

for area, users in all_users.items():
    for user in users:
        # print(user.get_importance())
        a_const = random.uniform(0.05,0.57)
        # print("Pns:", a_const)
        for server in servers:
            user.util_fun[server.num][0] = h(0.5,a_const)   # Pns

            imp = user.get_importance()[server.num]

            # Fn
            a = random.uniform(max(0.57*(math.sqrt(imp))-0.1, 0.05), 0.57*math.sqrt(imp))
            # print("Fn:", a)
            user.util_fun[server.num][1] = h(imp,a)   # Fn

            # Dn
            a = random.uniform(max(0.57*(math.sqrt(imp))-0.1, 0.05), 0.57*math.sqrt(imp))
            # print("Dn:", a)
            user.util_fun[server.num][2] = h(imp,a)   # Dn

# =================================================================== #

urban_servers = copy.deepcopy(servers)
subruban_servers = copy.deepcopy(servers)
rural_servers = copy.deepcopy(servers)

all_servers = {'Urban': urban_servers, 'Suburban': subruban_servers, 'Rural': rural_servers}

# ============================== Game Theory Matching ============================== #

# ============================== Approximate Matching ============================== #

gt_start = time.time()

gt_all_users = copy.deepcopy(all_users)
gt_all_servers = copy.deepcopy(all_servers)

for area, gt_users in gt_all_users.items(): 
    for user in gt_users:
        # For each user in the GT version, initialize its utility values
        user.current_ptrans = Ptrans_max
        user.current_fn = fn_max
        user.used_datasize = user.datasize

    for user in gt_users:
        # Now that the values are set, calculate the external magnitudes for all users
        user.set_magnitudes(gt_all_servers[area])
        

for area, gt_users in gt_all_users.items(): 

    if area != 'Urban':
        continue

    gt_servers = gt_all_servers[area]

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


# ============================== Regret Learning Matching - Complete Information ============================== #

regret_start = time.time()

regret_all_users = copy.deepcopy(all_users)
regret_all_servers = copy.deepcopy(all_servers)

for area, regret_users in regret_all_users.items(): 

    if area != 'Urban':
        continue

    regret_servers = regret_all_servers[area]

    regret_matching(regret_users, regret_servers)

    print("REMORSE Matching:\n")

    for u in regret_users:
        allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
        print("User:", u.num, "Server:", allegiance_num, "Ptrans:", u.current_ptrans/2, "Fn:", u.current_fn/(2 * 10**9), "Dn:", u.used_datasize/(u.datasize))

    print()

    regret_end = time.time()

    print("REMORSE Matching took", regret_end-regret_start, "seconds\n")

# =============================================================================== #

# ============================== Regret Learning Matching - Incomplete Information ============================== #

regret_start = time.time()

regret_all_users = copy.deepcopy(all_users)
regret_all_servers = copy.deepcopy(all_servers)

for area, regret_users in regret_all_users.items(): 

    if area != 'Urban':
        continue

    regret_servers = regret_all_servers[area]

    regret_matching_II(regret_users, regret_servers)

    print("REMORSE Matching II:\n")

    for u in regret_users:
        allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
        print("User:", u.num, "Server:", allegiance_num, "Ptrans:", u.current_ptrans/2, "Fn:", u.current_fn/(2 * 10**9), "Dn:", u.used_datasize/(u.datasize))

    print()

    regret_end = time.time()

    print("REMORSE Matching II took", regret_end-regret_start, "seconds\n")
    
# =============================================================================== #