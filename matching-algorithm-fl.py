
# Project description

# Imports
from Classes.User import User
from Classes.Server import Server
from Classes.CriticalPoint import CP
from approximate_matching import approximate_fedlearner_matching
from accurate_matching import accurate_fedlearner_matching
from Data.federated_learning import Servers_FL

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
    p = random.randint(4,6)
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
    

# ===================== Initialization of Users' Utility Values ===================== #
    
# Normalized Data Importance

for i in range(N):
    max_dist = 0
    for j in range(K):          # Finding Max Distance for each user
        user = users[i]
        user_x, user_y, user_z = user.x, user.y, user.z

        cp = critical_points[j]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

        if distance > max_dist:
            max_dist = distance

    for j in range(K):          # Calculating Importance for each user
        user = users[i]
        user_x, user_y, user_z = user.x, user.y, user.z

        cp = critical_points[j]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)   

        importance = distance / max_dist

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

# Normalized User Payment: Each server pays a random predetermined value between 1 and 10 to each of its users

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

# Normalized Data Quality: We assume all users have the same Dn

for i in range(K):

    # Finding Max Data Quality
    max_dq = 0
    for j in range(N):
        user = users[j]
        data_importance = user.get_importance()
        dq = data_importance[i]
        if (dq > max_dq):
            max_dq = dq

    # Calculating Normalized Data Quality
    for j in range(N):
        user = users[j]
        data_importance = user.get_importance()
        dq = data_importance[i]
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