import random 
from config import *

disasters = ["fire", "flood", "earthquake"]

# Data Rate
B = 10**7
I0 = 10**(-174 / 10)
P = [[0.25 for _ in range(N)] for _ in range(S)]
random_matrix = [[1 for _ in range(N)] for _ in range(S)]

# Energy Consumption
Z = [0] * N     
for i in range(N):
    Z[i] = 28.1 * random.uniform(0.95,1.05) * 10**3