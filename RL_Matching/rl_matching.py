from typing import List
from Classes.User import User
from Classes.Server import Server
from GT_Matching.utility_functions import *
from GT_Matching.accurate_matching_conditions import *
from RL_Matching.Action import Action
from general_parameters import *
import math
from pprint import pprint
import random

c = 1   # degree of exploration
Nt = [[0] * (S+1) for _ in range(N)]
rewards = [[0] * S for _ in range(N)]

def rl_fedlearner_matching(original_users: List[User], original_servers: List[Server]):
    t = 1

    users = original_users.copy()
    servers = original_servers.copy()

    for s in servers:
        print("Server", s.num, " Resources: ", s.p)
    
    while(t <= 1000*N):
        for u in users:     # For each user get its available actions
            actions = []
            if u.get_alligiance() is None:      # If it belongs to no server 
                for s in servers:
                    if s.Ns_max >= len(s.get_coalition()) + 1:     
                        action = Action(s)        # add all the available servers if they have the space
                        actions.append(action)
                actions.append(Action(None))         # and the option to stay out

            else:                                   # Else if user belongs to a server
                current_server = u.get_alligiance()
                for s in servers:                   # For the rest of the servers
                    if s != current_server and s.Ns_max >= len(s.get_coalition()) + 1:  # Add them as options if they have the space
                        action = Action(s)
                        actions.append(action)
                    elif s != current_server and s.Ns_max < len(s.get_coalition()) + 1: # For the servers that do not have space check for exchange
                        for other_user in list(s.get_coalition()):
                            action = Action(s, other_user)
                            actions.append(action)
                actions.append(Action(current_server))    # Add the option to stay to the current server
                actions.append(Action(None))              # And add the option to leave the server

            # After finding all the possible actions find the best one
            best_action, max_utility = choose_best_action(actions, u, servers, t)     

            # And execute it
            execute_action(best_action, max_utility, u)

        t += 1 
    pprint(Nt)
    print("\nRewards:")
    pprint(rewards)
    final_matching(original_users, original_servers)      
       
                        
def choose_best_action(actions, user: User, servers: List[Server], t):
    max_utility_diff = None
    best_action = None

    for a in actions:
        target = a.target
        other_user = a.exchange_user

        if target == None:      # If the target is to leave calculate the utility difference
            if user.get_alligiance() is not None:
                utility_diff = check_user_leaves_server(servers, user, user.get_alligiance())
                utility_diff = UCB_calc(utility_diff, a, t, user)        # calculate the utility based on the UCB algorithm
                if max_utility_diff is None or max_utility_diff < utility_diff:
                    best_action = a
                    max_utility_diff = utility_diff
            else:           # If the target is to stay out (utility = 0)
                utility_diff = 0
                if max_utility_diff is None or max_utility_diff < utility_diff:
                    best_action = a
                    max_utility_diff = utility_diff

        else:       # The user heads to a server
            if other_user == None:  # If there is no exchange
                if user.get_alligiance() == None:   # If user was not in a server
                    utility_diff = check_user_joins_server(servers, user, target)
                    utility_diff = UCB_calc(utility_diff, a, t, user)        # calculate the utility based on the UCB algorithm
                    if max_utility_diff is None or max_utility_diff < utility_diff:
                        best_action = a
                        max_utility_diff = utility_diff
                
                else:       # Else user belonged to a server and is changing (leaving from current to go to target)
                    if target != user.get_alligiance(): # if the user doesn't stay where he is
                        utility_diff = check_user_changes_servers(servers, user, user.get_alligiance(), target)
                        utility_diff = UCB_calc(utility_diff, a, t, user)        # calculate the utility based on the UCB algorithm
                        if max_utility_diff is None or max_utility_diff < utility_diff:
                            best_action = a
                            max_utility_diff = utility_diff
                    else:       # else the user stays to the same server / Utility Diff = 0
                        utility_diff = UCB_calc(0, a, t, user)      # calculate the utility based on the UCB algorithm
                        if max_utility_diff is None or max_utility_diff < utility_diff:
                            best_action = a
                            max_utility_diff = utility_diff

            else:        # Else there is exchange happening
                utility_diff = check_user_exchange(servers, user, other_user, user.get_alligiance(), target)
                utility_diff = UCB_calc(utility_diff, a, t, user)        # calculate the utility based on the UCB algorithm
                if max_utility_diff is None or max_utility_diff < utility_diff:
                    best_action = a
                    max_utility_diff = utility_diff

    return best_action, max_utility_diff

def UCB_calc(utility, a, t, user: User):
    UCB_factor = 1  # set a big tarting UCB factor to explore all actions available
    if a.target != None and Nt[user.num][a.target.num] != 0:
        UCB_factor = c * math.sqrt(math.log(t)/Nt[user.num][a.target.num])
    elif a.target == None and Nt[user.num][S] != 0:
        UCB_factor = c * math.sqrt(math.log(t)/Nt[user.num][S])

    return utility + UCB_factor

def execute_action(action, max_utility, user: User):
    target = action.target
    other_user = action.exchange_user

    if target == None:      # If the target is to leave 
        current_server = user.get_alligiance()
        user.change_server(None)
        if current_server is not None:     # if user is not already out
            current_server.remove_from_coalition(user)
        Nt[user.num][S] += 1    # Update Nt 

    else:       # The user heads to a server
        if other_user == None:  # If there is no exchange
            if user.get_alligiance() == None:   # If user was not in a server and joins
                user.change_server(target)
                target.add_to_coalition(user)
                     
            else:       # Else user belonged to a server and is changing (leaving from current to go to target)
                current_server = user.get_alligiance()
                current_server.remove_from_coalition(user)      # remove from current server
                user.change_server(target)
                target.add_to_coalition(user)                   # and add to the new

        else:        # Else there is exchange happening
            current_server = user.get_alligiance()

            current_server.remove_from_coalition(user)      # remove our user from current server
            user.change_server(target)
            target.add_to_coalition(user)                   # and add to the target

            target.remove_from_coalition(other_user)        # remove the exchange user from target
            other_user.change_server(current_server)
            current_server.add_to_coalition(other_user)     # and add to our previously current server

        Nt[user.num][target.num] += 1   # Update Nt
        rewards[user.num][target.num] += max_utility  # and add the reward our environment gets
        


def final_matching(users: List[User], servers: List[Server]):
    flag = True     # flag that checks change - if no change the end
    
    while(flag):
        flag = False
        random.shuffle(servers)     # shuffle servers each iteration to not prioritize any server
        for server in servers:
            random.shuffle(users)   # shuffle users each iteration to not prioritize any user
            for user in users:
                user_reward = rewards[user.num][server.num]

                # if user belongs to None and there is space
                if user.get_alligiance() is None and len(server.get_coalition()) < server.Ns_max:
                    if user_reward > 0:     # if user reward > 0
                        user.change_server(server)      # add the user to the server
                        server.add_to_coalition(user)
                        flag = True         # and set flag to True (change happened)

                # else if the user belongs to a server and there is space
                elif user.get_alligiance() is not None and user.get_alligiance() != server and len(server.get_coalition()) < server.Ns_max:
                    if user_reward > rewards[user.num][user.get_alligiance().num]:  # if this server and user benefit more
                        user.get_alligiance().remove_from_coalition(user)   # remove from its original server and
                        user.change_server(server)      # add the user to the server
                        server.add_to_coalition(user)
                        flag = True         # and set flag to True (change happened)

                # else if the user belongs to None and there is no space
                elif user.get_alligiance() is None and len(server.get_coalition()) == server.Ns_max:
                    u_min_contribute = None
                    for u in server.get_coalition():    # check for the minimum contributing user in the current coalition if the server can benefit from exchange
                        if u_min_contribute is None or rewards[u_min_contribute.num][server.num] > rewards[u.num][server.num]:
                            u_min_contribute = u

                    if user_reward > rewards[u_min_contribute.num][server.num]:        # if it can
                        server.remove_from_coalition(u_min_contribute)             # remove the other user
                        u_min_contribute.change_server(None)

                        user.change_server(server)                  # and add our user
                        server.add_to_coalition(user)
                        flag = True     # and set flag to True (change happened)

                # else if the user belongs to another server and there is no space
                elif user.get_alligiance() is not None and user.get_alligiance() != server and len(server.get_coalition()) == server.Ns_max:
                    u_min_contribute = None
                    for u in server.get_coalition():    # check for the minimum contributing user in the current coalition if the server can benefit from exchange
                        if u_min_contribute is None or rewards[u_min_contribute.num][server.num] > rewards[u.num][server.num]:
                            u_min_contribute = u

                    # if it can and user wants
                    if user_reward > rewards[u_min_contribute.num][server.num] and user_reward > rewards[user.num][server.num]:        
                        server.remove_from_coalition(u_min_contribute)                 # remove the other user
                        u_min_contribute.change_server(user.get_alligiance())
                        user.get_alligiance().add_to_coalition(u_min_contribute)       # and add him to the other server

                        user.get_alligiance().remove_from_coalition(user)   # and remove user from its coalition
                        user.change_server(server)
                        server.add_to_coalition(user)                   # and add him to the server
                        flag = True     # and set flag to True (change happened)

    users.sort(key=lambda x: x.num)     # sort users by number



