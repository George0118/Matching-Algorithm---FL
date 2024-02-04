import random 

# Parameters

N = 10  # Users -- min value: 5
S = 3   # Servers
K = 3   # Critical Points

disasters = ["fire", "flood", "earthquake"]

# Data Rate
I0 = 10 ** ((-104 - 30) / 10)
P = [[random.uniform(0.7, 1) for _ in range(N)] for _ in range(S)]

# Energy Consumption
Z = [0] * N     
for i in range(N):
    Z[i] = 28.1 * random.uniform(0.95,1.05) * 10**3