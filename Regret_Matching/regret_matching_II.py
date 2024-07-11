from Regret_Matching.utility_function import user_utility_ext
from Classes.User import User, quantized_fn, quantized_datasize, quantized_ptrans
from Classes.Server import Server
from typing import List
from Regret_Matching.Action import Action
from config import N
import numpy as np
import itertools
import random
import copy

convergence_limit = 0.5

# White Noise
def white_noise():
    wn = np.random.normal(loc=0.0, scale=0.2, size=None)
    return wn

# Learning rate functions
def l(t):
    return 1 / t

def g(t):
    return 1 / (t * np.log(t+1))

def m(t):
    return 1 / (t * (np.log(t+1))**2)

# Function to create all possible actions of a user
def matching_actions(servers, quantized_fn, quantized_ptrans, quantized_datasize):
    # Compute the Cartesian product of the input lists
    all_combinations = list(itertools.product(servers, quantized_fn, quantized_ptrans, quantized_datasize))
    all_combinations = [Action(a[0], a[1], a[2], a[3]) for a in all_combinations]
    return all_combinations


def regret_matching_II(users: List[User], servers: List[Server], epsilon = 0):
    t = 1
    actions = matching_actions(servers+[None], quantized_fn, quantized_ptrans, quantized_datasize)

    # Probabilities Initialization
    user_probabilities = [1/((len(servers)+1) * len(quantized_fn) * len(quantized_ptrans) * len(quantized_datasize))] * ((len(servers)+1) * len(quantized_fn) * len(quantized_ptrans) * len(quantized_datasize))
    probabilities = []
    for _ in range(N):
        probs = copy.deepcopy(user_probabilities)
        probabilities.append(probs)
    
    # Regrets Initialization
    regret_vector = []
    for _ in range(N):
        regret_vector.append([0]*len(user_probabilities))

    # Utilities Initialization
    utilities_vector = []
    for _ in range(N):
        utilities_vector.append([0]*len(user_probabilities))

    actions_taken = [None] * len(users)
    while(not convergence(probabilities) and t < 5000):
        print("t =",t)
        for u in users:     # For each user get its most likely action
            action_index, repeat = select_action(probabilities[u.num])
            if not repeat:
                execute_action(u, servers, actions[action_index])
                actions_taken[u.num] = actions[action_index]

        # Calculate magnitudes based on the actions taken from each user to use in updating regret
        for u in users:
            u.set_magnitudes(servers)

        # Calculate current utilities
        current_utilities = [0] * len(users)
        for u in users:
            current_utilities[u.num] = user_utility_ext(u, u.get_alligiance()) + white_noise()

        update_utilities_vector(utilities_vector, users, actions, current_utilities, t) # Update utilities vector
        update_regret_vector(regret_vector, utilities_vector, users, actions, current_utilities, t)  # Update regret vector
        update_probabilities(probabilities, regret_vector, users, actions, t)   # Update probabilities

        for u in users:
            if not all(p == 0.0 for p in probabilities[u.num]):
                max_value = max(regret_vector[u.num])
                max_index = regret_vector[u.num].index(max_value)
                print("User", u.num, ", best server:", actions[max_index].target.num, ", Fn:", actions[max_index].fn, ", Dn:", actions[max_index].ds, ", PTrans:", actions[max_index].ptrans,", max prob:", probabilities[u.num][max_index])
            else:
                print("User", u.num, ", best server:", actions_taken[u.num].target.num, ", current server:", u.get_alligiance().num if u.get_alligiance() is not None else None)

        t += 1 

    # After convergence execute the best action for each user
    for user in users:
        if not all(p == 0 for p in probabilities[user.num]):
            max_value = max(probabilities[user.num])
            max_index = probabilities[user.num].index(max_value)
            execute_action(user, servers, actions[max_index])
    for user in users:
            user.set_magnitudes(servers)


# Convergence Check
def convergence(probabilities: List[List]):
    for user_probabilities in probabilities:
        prob_sum = sum(user_probabilities)
        found_converging_value = False

        if prob_sum == 0:    # If sum is 0, then the user has converged
            continue
        
        for prob in user_probabilities:
            percentage = prob/prob_sum
            if percentage > convergence_limit:    # Else if you find dominant value the user has converged
                found_converging_value = True

        if found_converging_value == False: # If not then we haven't converged
            return False
        
    return True # If for all users we found dominant values we have converged
                

# Select Action based on probabilities
def select_action(probabilities):
    # If all probabilities for the user are zero, means we chose the best action and
    # except if something changes we have no other reason to select something else
    repeat = all(p == 0 for p in probabilities)
    indices = list(range(len(probabilities)))
    selected_index = random.choices(indices, weights=probabilities, k=1)[0]
    return selected_index, repeat


# Execute selected Action
def execute_action(user: User, servers: List[Server], action: Action, update_magnitudes = False, external_denominators = None):
    # Matching the server_target of the action to the one of the copied servers
    if action.target is not None:
        actual_target = servers[action.target.num]
    else:
        actual_target = None

    # Change the user's server and make the necessary changes to the server it left from or went to
    if actual_target != user.get_alligiance():
        if user.get_alligiance() is not None:
            user.get_alligiance().remove_from_coalition(user.num)

        if actual_target is not None:
            user.change_server(actual_target)
        else:
            user.change_server(None)

        if actual_target is not None:
            actual_target.add_to_coalition(user)

    # Update the user's parameters and update the changes in magnitudes only for the target server
    user.set_parameters(action.fn, action.ptrans, action.ds)
    if update_magnitudes:
        if action.target is not None:
            user.set_magnitudes(servers, change_all=False, server_num=action.target.num, external_denominator=external_denominators[action.target.num])


# Update Utilities Vector
def update_utilities_vector(utilities_vector: List[List], users: List[User], actions: List[Action], current_utilities: List, t):
    for user in users:
        # For each user get its current utility 
        current_utility = current_utilities[user.num]
        for action_index, action in enumerate(actions):
            new_utility_term = l(t) * (current_utility - utilities_vector[user.num][action_index])
            utilities_vector[user.num][action_index] += new_utility_term


# Update Regret Vector
def update_regret_vector(regret_vector: List[List], utilities_vector: List[List], users: List[User], actions: List[Action], current_utilities: List, t):
    for user in users:
        # For each user get its current utility
        current_utility = current_utilities[user.num]
        for action_index, action in enumerate(actions):
            new_regret_term = g(t) * (utilities_vector[user.num][action_index] - current_utility - regret_vector[user.num][action_index])
            regret_vector[user.num][action_index] += new_regret_term


k_m = 1
# Boltzmann-Gibbs
def boltzmann_gibbs(user_regret_vector: List):
    sum_value = 0
    for regret in user_regret_vector:
        sum_value += np.exp(max(0,regret)/k_m)

    bg_vector = []
    for regret in user_regret_vector:
        bg_vector.append(np.exp(max(0,regret)/k_m)/sum_value)

    return bg_vector

# Update Probabilities
def update_probabilities(probabilities: List[List], regret_vector: List[List], users: List[User], actions: List[Action], t):
    for user in users:
        # For each user get its boltzmann_gibbs vector
        bg_vector = boltzmann_gibbs(regret_vector[user.num]) 
        for action_index, action in enumerate(actions):
            new_probability_term = m(t) * (bg_vector[action_index] - probabilities[user.num][action_index])
            print(new_probability_term)
            probabilities[user.num][action_index] += new_probability_term