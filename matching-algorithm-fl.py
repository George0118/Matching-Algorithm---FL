
# Project description

# Imports
from Classes.User import User
from Classes.Server import Server
from Classes.CriticalPoint import CP
from approximate_matching import approximate_fedlearner_matching
from accurate_matching import accurate_fedlearner_matching
from Data.federated_learning import Servers_FL
from Data.load_images import fire_input_paths, flood_input_paths, earthquake_input_paths, count_images, factor
import os

# General Parameters

N = 10  # Users -- min value: 5
S = 3   # Servers
K = 3   # Critical Points


# ===================== Users', Servers' and Critical Points' Topology ===================== #

import random
import math

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
disasters = ["fire", "flood", "earthquake"]

for i in range(K):
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    y = random.uniform(-1,1)
    disaster = disasters[i%3]
    
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

# ========================================================================================== #  
        
print()

# ===================== Initialization of Users' Utility Values ===================== #
        
# Data size for each user
        
Dn = [0]*N
        
for s in servers:   # For each server(disaster) calculate number of images each user will receive
    cps = s.get_critical_points()   # Get the relevant Critical Points
    user_min_distances = [-1]*N
    sizes = [0]*N

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

    # Calculate the data sizes based on the user minimum distance from the CPs
    for i in range(N):
      sizes[i] = int(image_num*math.sqrt(1/user_min_distances[i])/N)

    # Normalize them
    total_size = sum(sizes)
    sizes = [size * image_num // total_size for size in sizes]

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

# Assumptions: No noise, channel gain is inversely proportional with the distance between user and server, transmit power is the same for all users and their servers thus is neglected, bandwidth 50Mbps

I0 = ((10 * 10**6) * 3.981 * 10**(-21))/N

P = [[random.uniform(0.7, 1) for _ in range(N)] for _ in range(S)]

# Finding Max Data Rate
max_dr = 0
for i in range(N):
    for j in range(S):
        user = users[i]
        user_x, user_y, user_z = user.x, user.y, user.z

        server = servers[j]
        server_x, server_y, server_z = server.x, server.y, server.z

        distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)
        
        g = 1/distance
        
        dr = math.log2(1 + g*P[j][i]/I0)
        
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
        
        g = 1/distance
        
        dr = math.log2(1 + g*P[j][i]/I0)
        
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
        
        g = 1/distance
        
        dr = math.log2(1 + g*P[j][i]/I0)

        Z = 28.1 * random.uniform(0.95,1.05) * 10**3

        E_transmit = Z*P[j][i]/dr

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
        
        g = 1/distance
        
        dr = math.log2(1 + g*P[j][i]/I0)

        Z = 28.1 * random.uniform(0.95,1.05) * 10**3

        E_transmit = Z*P[j][i]/dr

        user.add_Etransmit(E_transmit/max_E_transmit)


# ==================================================================

# Normalized Data Quality

for i in range(K):

    # Finding Max Data Quality
    max_dq = 0
    for j in range(N):
        user = users[j]
        data_importance = user.get_importance()
        dq = (2 - data_importance[i]) * user.get_datasize()
        if (dq > max_dq):
            max_dq = dq

    # Calculating Normalized Data Quality
    for j in range(N):
        user = users[j]
        data_importance = user.get_importance()
        dq = (2 - data_importance[i]) * user.get_datasize()
        user.add_dataquality(dq/max_dq)        

# =================================================================================== #


# ============================== Approximate Matching ============================== #

# Initializing the available servers for each user
for i in range(N):
    u = users[i]
    u.set_available_servers(servers)
    
approximate_fedlearner_matching(users, servers)

print("Approximate FedLearner Matching:\n")

for u in users:
    allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
    print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

print()
# =================================================================================== #


# ============================== Accurate Matching ============================== #
    
accurate_fedlearner_matching(users, servers)

print("Accurate FedLearner Matching:\n")

for u in users:
    allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
    print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

print()
# =============================================================================== #

rounds=100 # number of global rounds
lr=0.001 # learning rate
epoch=1 # local iterations

server_losses, server_accuracy = Servers_FL(users, servers, rounds, lr, epoch)