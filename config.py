import argparse

def parse_arguments():
    global N, K, num

    # Parse Arguments (Number of Users and Number of Critical Points)
    parser = argparse.ArgumentParser(description="Matching Algorithms - Federated Learning")
    parser.add_argument("--a", type=int, help="Number of run")
    parser.add_argument("--n", type=int, default=10, help="Number of Users")
    parser.add_argument("--k", type=int, default=3, help="Number of Critical Points")
    parse = parser.parse_args()

    num =parse.a
    N = parse.n
    K = parse.k

# Parameters

num = 1 # Number of run
N = 0   # Users 
S = 3   # Servers
K = 0   # Critical Points

thread_count = 5