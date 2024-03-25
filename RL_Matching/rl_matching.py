from typing import List
from Classes.User import User
from Classes.Server import Server
from GT_Matching.utility_functions import *
from GT_Matching.accurate_matching_conditions import *
from RL_Matching.Action import Action
from config import N,S
import math
from pprint import pprint
import random
import copy
from itertools import product

rew_for_out = 1e-6
g = 0.5
c = 2   # degree of exploration
Nt = [[0] * (S+1) for _ in range(N)]
Qn = [[0] * (S) for _ in range(N)]

def rl_fedlearner_matching(original_users: List[User], original_servers: List[Server], server_focused):
    global Nt, Qn

    t = 1

    flag = False
    prev_best_action = [None]*N
    same_target_count = [0]*N

    convergence = False

    users = copy.deepcopy(original_users)
    servers = copy.deepcopy(original_servers)

    # Set everything to 0 in case the global array its already changed from before
    Nt = [[0] * (S+1) for _ in range(N)]
    Qn = [[0] * (S+1) for _ in range(N)]
    
    while(not convergence and not flag):
        random.shuffle(users)
        for u in users:     # For each user get its available actions
            actions = []
            if u.get_alligiance() is None:      # If it belongs to no server 
                for s in servers:               # add all the available servers 
                    if s.Ns_max > len(s.get_coalition()):   # if there is space
                        action = Action(s)        
                        actions.append(action)
                    else:                       # else if there is not
                        for other_user in s.get_coalition():      # add all exchanges with users of the coalition
                            action = Action(s,other_user)
                            actions.append(action)

                actions.append(Action(None))         # and the option to stay out

            else:                                   # Else if user belongs to a server
                current_server = u.get_alligiance()
                for s in servers:                   # For the rest of the servers
                    if s != current_server and s.Ns_max > len(s.get_coalition()):  # Add them as options if they have the space
                        action = Action(s)
                        actions.append(action)
                    elif s != current_server and s.Ns_max <= len(s.get_coalition()): # For the servers that do not have space check for exchange
                        for other_user in list(s.get_coalition()):
                            action = Action(s, other_user)
                            actions.append(action)
                actions.append(Action(current_server))    # Add the option to stay to the current server
                actions.append(Action(None))              # And add the option to leave the server

            # After finding all the possible actions find the best one
            best_action, max_reward = choose_best_action(actions, u, servers, t, server_focused)  

            if(t>1):
                if prev_best_action[u.num] == best_action.target:
                    same_target_count[u.num] += 1
                else:
                    same_target_count[u.num] = 0
            
            prev_best_action[u.num] = best_action.target

            # And execute it
            execute_action(best_action, max_reward, u)

        if t != 1:
            convergence = check_convergence(prev_Qn, Qn)

        if t > 5:
            flag = True
            for u in users:
                if same_target_count[u.num] < 5:
                    flag = False
                    break

        prev_Qn = copy.deepcopy(Qn)
        t += 1 

    print("\nQn:")
    pprint(Qn)
    final_matching(original_users, original_servers)  
    print()    
       
                        
def choose_best_action(actions, user: User, servers: List[Server], t, server_focused):
    max_Qn = None
    best_action = None
    reward = None
    max_utility = None

    for a in actions:
        target = a.target

        if target is None:      # If the target is to leave or stay out
            if server_focused:
                utility = rew_for_out
            else:
                utility = rew_for_out

            if max_utility is None or max_utility < utility:
                max_utility = utility

            Q = Qn[user.num][S]

            Q = UCB_calc(Q, a, t, user)        # calculate the new Q based on the UCB algorithm
            if max_Qn is None or max_Qn < Q:
                best_action = a
                max_Qn = Q
                reward = utility

        else:       # The user heads to a server 
            if server_focused:
                new_coalition = target.get_coalition().union({user})
                utility = server_utility_externality(servers, new_coalition, target)/len(list(new_coalition))
            else:
                utility = user_utility_ext(user, target)

            if max_utility is None or max_utility < utility:
                max_utility = utility

            Q = Qn[user.num][target.num]

            Q = UCB_calc(Q, a, t, user)        # calculate the new Q based on the UCB algorithm
            if max_Qn is None or max_Qn < Q:
                best_action = a
                max_Qn = Q
                reward = utility

    reward = reward/max_utility

    return best_action, reward

def UCB_calc(utility, a, t, user: User):
    UCB_factor = 5  # set a big starting UCB factor to explore all actions available

    if a.target != None and Nt[user.num][a.target.num] != 0:
        UCB_factor = c * math.sqrt(math.log(t)/Nt[user.num][a.target.num])
    elif a.target is None and Nt[user.num][S] != 0:
        UCB_factor = c * math.sqrt(math.log(t)/Nt[user.num][S])

    return utility + UCB_factor

def execute_action(action, reward, user: User):
    target = action.target
    other_user = action.exchange_user

    if target is None:      # If the target is to leave 
        current_server = user.get_alligiance()
        user.change_server(None)
        if current_server is not None:     # if user is not already out
            current_server.remove_from_coalition(user.num)
        Nt[user.num][S] += 1    # Update Nt
        Qn[user.num][S] += g * (reward - Qn[user.num][S]) 

    else:       # The user heads to a server
        if other_user is None:  # If there is no exchange
            if user.get_alligiance() is None:   # If user was not in a server and joins
                user.change_server(target)
                target.add_to_coalition(user)
                     
            else:       # Else user belonged to a server and is changing (leaving from current to go to target)
                current_server = user.get_alligiance()
                current_server.remove_from_coalition(user.num)      # remove from current server
                user.change_server(target)
                target.add_to_coalition(user)                   # and add to the new

        else:        # Else there is exchange happening
            if user.get_alligiance() is None:
                target.remove_from_coalition(other_user.num)        # remove the exchange user from target's coalition
                other_user.change_server(None)
                target.add_to_coalition(user)                   # add the user to the target's coalition
                user.change_server(target)

            else:
                current_server = user.get_alligiance()

                current_server.remove_from_coalition(user.num)      # remove our user from current server
                user.change_server(target)
                target.add_to_coalition(user)                   # and add to the target

                target.remove_from_coalition(other_user.num)        # remove the exchange user from target
                other_user.change_server(current_server)
                current_server.add_to_coalition(other_user)     # and add to our previously current server

        Nt[user.num][target.num] += 1    # Update Nt
        Qn[user.num][target.num] += g * (reward - Qn[user.num][target.num]) 
        


def final_matching(users: List[User], servers: List[Server]):
    flag = True     # flag that checks change - if no change the end

    cartesian_product = list(product(servers, users))
    
    while(flag):
        flag = False
        random.shuffle(cartesian_product)     # shuffle cartesian product each iteration to not prioritize any server, user
        for server, user in cartesian_product:
            user_reward = Qn[user.num][server.num]

            # print("Server:", server.num, " | User:", user.num)

            current_server = None
            if user.get_alligiance() is not None:
                for s in servers:
                    if s.num == user.get_alligiance().num:
                        current_server = s
                        break

            # print(f"Current Server of User {user.num} is {current_server.num if current_server is not None else -1}")

            # if user belongs to None and there is space
            if current_server is None and len(server.get_coalition()) < server.Ns_max:
                if user_reward > 0:     # if user reward > 0
                    user.change_server(server)      # add the user to the server
                    server.add_to_coalition(user)
                    flag = True         # and set flag to True (change happened)

            # else if the user belongs to a server and there is space
            elif current_server is not None and current_server.num != server.num and len(server.get_coalition()) < server.Ns_max:
                if user_reward > Qn[user.num][user.get_alligiance().num]:  # if this server and user benefit more
                    current_server.remove_from_coalition(user.num)   # remove from its original server and
                    user.change_server(server)      # add the user to the server
                    server.add_to_coalition(user)
                    flag = True         # and set flag to True (change happened)
                    # print("Changed and server had space")

            # else if the user belongs to None and there is no space
            elif current_server is None and len(server.get_coalition()) == server.Ns_max:
                u_min_contribute = None
                for u in server.get_coalition():    # check for the minimum contributing user in the current coalition if the server can benefit from exchange
                    if u_min_contribute is None or Qn[u_min_contribute.num][server.num] > Qn[u.num][server.num]:
                        for u1 in users:
                            if u1.num == u.num:
                                u_min_contribute = u1
                                break

                if user_reward > Qn[u_min_contribute.num][server.num]:        # if it can
                    server.remove_from_coalition(u_min_contribute.num)             # remove the other user
                    u_min_contribute.change_server(None)

                    user.change_server(server)                  # and add our user
                    server.add_to_coalition(user)
                    flag = True     # and set flag to True (change happened)
                    # print("There was exchange between None and Server with min contributing User", u_min_contribute.num)

            # else if the user belongs to another server and there is no space
            elif current_server is not None and current_server.num != server.num and len(server.get_coalition()) == server.Ns_max:
                u_min_contribute = None
                for u in server.get_coalition():    # check for the minimum contributing user in the current coalition if the server can benefit from exchange
                    if u_min_contribute is None or Qn[u_min_contribute.num][server.num] > Qn[u.num][server.num]:
                        for u1 in users:
                            if u1.num == u.num:
                                u_min_contribute = u1
                                break

                # if it can and user wants
                if user_reward > Qn[u_min_contribute.num][server.num] and user_reward > Qn[user.num][user.get_alligiance().num]:        
                    server.remove_from_coalition(u_min_contribute.num)                 # remove the other user
                    u_min_contribute.change_server(current_server)
                    current_server.add_to_coalition(u_min_contribute)       # and add him to the other server

                    current_server.remove_from_coalition(user.num)   # and remove user from its coalition
                    user.change_server(server)
                    server.add_to_coalition(user)                   # and add him to the server
                    flag = True     # and set flag to True (change happened)
                    # print("There was exchange with min contributing User", u_min_contribute.num)

            
            if user_reward <= 0 and user.get_alligiance() is not None and user.get_alligiance().num == server.num:     # if there is no benefit the user goes unmatched
                for s in servers:
                    if s.num == user.get_alligiance().num:
                        current_server = s
                        break
                current_server.remove_from_coalition(user.num)
                user.change_server(None)
                flag = True
                # print("User", user.num, "got removed from server", current_server.num)


    users.sort(key=lambda x: x.num)     # sort users by number
    servers.sort(key=lambda x: x.num)     # sort servers by number


def check_convergence(prev_Qn, Qn):
    return are_lists_equal(prev_Qn, Qn)

def are_lists_equal(list1, list2, tolerance=rew_for_out):
    # Check if the lists have the same dimensions
    if len(list1) != len(list2):
        return False
    
    # Iterate over corresponding elements in each list
    for i in range(len(list1)):
        # If the elements are lists themselves, recursively call the function
        if isinstance(list1[i], list) and isinstance(list2[i], list):
            if not are_lists_equal(list1[i], list2[i], tolerance):
                return False
        # If the elements are not lists, check for equality within the tolerance
        elif abs(list1[i] - list2[i]) > tolerance:
            return False
    
    # If all corresponding elements are within the tolerance, the lists are considered equal
    return True

