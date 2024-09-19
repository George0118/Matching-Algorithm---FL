import random 
from config import *

disasters = ["fire", "flood", "earthquake"]

N_max = 100 # At max set 100 users
current_max_users = 30  # For our experiments we have max 30 users
N_neutral = 250 # Neutral images per user

# Data Rate
B = 10**7
I0 = 10**(-174 / 10)
P = [[0.25 for _ in range(N)] for _ in range(S)]
random_matrix = [[random.uniform(0,1) for _ in range(N)] for _ in range(S)]

# Bits to transfer
Z = 28.1 * 10**3