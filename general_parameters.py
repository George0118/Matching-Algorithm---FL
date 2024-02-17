import random 
from config import *

disasters = ["fire", "flood", "earthquake"]

# Data Rate
I0 = 10 ** ((-104 - 30) / 10)
P = [[round(random.uniform(0.01, 1), 2) for _ in range(N)] for _ in range(S)]
random_matrix = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(S)]

# Energy Consumption
Z = [0] * N     
for i in range(N):
    Z[i] = 28.1 * random.uniform(0.95,1.05) * 10**3