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
from helping_functions import dataset_sizes
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

urban_threshold = 30
suburban_threshold = 21
rural_threshold = 12
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

distance_diff = 0.025

for i in range(min(rural_threshold, N)):

    while True:
        j = i//K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if j == 0 and distance >= 0.02 and distance < 0.025:
            break
        if j != 0 and distance < 0.05 and distance >= 0.025:  # if in the desired sphere and good user then add the user
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

        if distance < 0.05 and distance >= 0.025:  # if in the desired sphere and good user then add the user
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

        if distance > 0.3 and distance <= 0.4:     # if in the desired sphere and bad user then add the user
            break
        
    user = User(x,y,z,i+rural_threshold)
    
    rural_users.append(copy.deepcopy(user))


for i in range(min(urban_threshold - suburban_threshold, N - suburban_threshold)):

    while True:
        j = i/K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.05 and distance >= 0.025:  # if in the desired sphere and good user then add the user
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

        if distance > 0.2 and distance <= 0.3:     # if in the desired sphere and bad user then add the user
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

        if distance > 0.3 and distance <= 0.4:     # if in the desired sphere and bad user then add the user
            break
        
    user = User(x,y,z,i+suburban_threshold)
    
    rural_users.append(copy.deepcopy(user))

# ========================================================================================== #  
        
all_users = {'Urban': urban_users, 'Suburban': suburban_users, 'Rural': rural_users}
print()

# ===================== Initialization of Users' Utility Values ===================== #
        
# Data size for each user
        
for s in servers:   # For each server(disaster) calculate number of images each user will receive
    cps = s.get_critical_points()   # Get the relevant Critical Points

    total_images = count_images(fire_input_paths + flood_input_paths + earthquake_input_paths)

    # For each user calculate the minimum distance from the relevant Critical Points
    for area, users in all_users.items():
        Dn = [0]*N

        sizes = dataset_sizes(s, area, users, cps, total_images)

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

for area, users in all_users.items():
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

# Normalized Data Rate with Externality

# Finding Max Data Rate
for area, users in all_users.items():

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
for area, users in all_users.items():    
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

# Normalized Local Energy Consumption

# Finding Max Local Energy Consumption
for area, users in all_users.items(): 
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
        

# Normalized Energy Consumption to transmit the local model parameters to the server with Externality

# Find Max Transmission Energy
for area, users in all_users.items(): 
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

for area, users in all_users.items(): 
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

for area, users in all_users.items():      
    for u in users:
        l=[]
        for s in servers:
            l.append(user_utility_ext(u,s,False))
        print(l) 
        
print()


urban_servers = copy.deepcopy(servers)
subruban_servers = copy.deepcopy(servers)
rural_servers = copy.deepcopy(servers)

all_servers = {'Urban': urban_servers, 'Suburban': subruban_servers, 'Rural': rural_servers}