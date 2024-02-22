import random
from Classes.Server import Server
from Classes.User import User
from typing import List
from GT_Matching.accurate_matching_conditions import *
from GT_Matching.utility_functions import *
from config import N,S
import copy
from pprint import pprint

#Accurate Matching Function

rewards = [[0] * S for _ in range(N)]

def accurate_fedlearner_matching(original_apprx_matched_users, original_servers):

    apprx_matched_users = copy.deepcopy(original_apprx_matched_users)
    servers = copy.deepcopy(original_servers)

    t = 1
    while(t <= 1000*N):

        random_user = apprx_matched_users[random.randint(0,N-1)]   # Select random user

        if(random_user.get_alligiance() == None):     # If user does not belong to any coalition
            max_utility_diff = 0
            favorite_MEC = None
            for s in servers:       # Find the server that has space and benefits more from him
                if len(s.get_coalition()) < s.Ns_max:
                    utility_diff = check_user_joins_server(servers,random_user,s)
                    rewards[random_user.num][s.num] += utility_diff
                    if utility_diff > max_utility_diff:
                        max_utility_diff = utility_diff
                        favorite_MEC = s

            # If there is a server that benefits from the user, add the user to its coalition
            if(favorite_MEC is not None):
                favorite_MEC.add_to_coalition(random_user)
                random_user.change_server(favorite_MEC)
                rewards[random_user.num][favorite_MEC.num] += max_utility_diff
                # print("Added user to server")
            else:       # Else check for full servers for exchange
                max_utility_diff = 0
                favorite_MEC = None
                exchange_user = None
                for s in servers:
                    if len(s.get_coalition()) == s.Ns_max:  # If the server is full check for exchange
                        for u in s.get_coalition():
                            utility_diff = check_user_exchange(servers, u, random_user, s, None)
                            if utility_diff > max_utility_diff:
                                max_utility_diff = utility_diff
                                favorite_MEC = s
                                exchange_user = u
                if(favorite_MEC is not None and exchange_user is not None):
                    favorite_MEC.remove_from_coalition(exchange_user)   # remove exchange user from server
                    exchange_user.change_server(None)
                    favorite_MEC.add_to_coalition(random_user)          # add random user to server
                    random_user.change_server(favorite_MEC)

                    rewards[random_user.num][favorite_MEC.num] += max_utility_diff
                    # print("User from None joins server")

        else:   # else the user belongs in a coalition already

            current_server = random_user.get_alligiance()

            other_server = random.choice(servers)      # select another server randomly
            while(other_server == current_server):     # different than the current server 
                other_server = random.choice(servers)

            if(len(other_server.get_coalition()) < other_server.Ns_max):      # If the other server has space
                utility_diff = check_user_changes_servers(servers, random_user, current_server, other_server)   # check if giving away a user is better
                
                if(utility_diff > 0):   # if it is better
                    current_server.remove_from_coalition(random_user)   # remove from current server
                    other_server.add_to_coalition(random_user)          # add to other server
                    random_user.change_server(other_server)
                    rewards[random_user.num][other_server.num] += utility_diff  
                    # print("User changed servers")   
                else:       # Reward the user for staying at the coalition
                    rewards[random_user.num][random_user.get_alligiance().num] -= utility_diff      

            else:   # if the other server has no space
                other_coalition = other_server.get_coalition()

                other_user = random.choice(list(other_coalition))  # select a random user from its coalition

                utility_diff = check_user_exchange(servers, random_user, other_user, current_server, other_server)   # check if the servers exchanging users is beneficial

                if(utility_diff > 0):   # if it is better
                    current_server.remove_from_coalition(random_user)   # remove random user from current server
                    other_server.add_to_coalition(random_user)          # add random user to other server
                    random_user.change_server(other_server)

                    other_server.remove_from_coalition(other_user)      # remove other user from other server
                    current_server.add_to_coalition(other_user)         # add other user to current server
                    other_user.change_server(current_server)

                    rewards[random_user.num][other_server.num] += utility_diff
                    rewards[other_user.num][current_server.num] += utility_diff
                
                    # print("User exchange")

                else:       # Reward the users for staying at their coalition
                    rewards[random_user.num][random_user.get_alligiance().num] -= utility_diff
                    rewards[other_user.num][other_server.num] -= utility_diff

        # Lastly check if removing the user from its current coalition would be beneficial
        if(random_user.get_alligiance() != None):
            current_server = random_user.get_alligiance()

            utility_diff = check_user_leaves_server(servers, random_user, current_server)

            if(utility_diff > 0):       # If the server benefits from excluding the user
                current_server.remove_from_coalition(random_user)   # remove from coalition
                random_user.change_server(None)
                # print("User removed from server")

        t += 1
        # print(rewards[0])
        # for s in servers:
        #     print(user_utility_ext(apprx_matched_users[0], s, True))

    print("\nRewards:")
    pprint(rewards)
    print()
    final_matching(original_apprx_matched_users, original_servers)


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
                    if user_reward > rewards[u_min_contribute.num][server.num] and user_reward > rewards[user.num][user.get_alligiance().num]:        
                        server.remove_from_coalition(u_min_contribute)                 # remove the other user
                        u_min_contribute.change_server(user.get_alligiance())
                        user.get_alligiance().add_to_coalition(u_min_contribute)       # and add him to the other server

                        user.get_alligiance().remove_from_coalition(user)   # and remove user from its coalition
                        user.change_server(server)
                        server.add_to_coalition(user)                   # and add him to the server
                        flag = True     # and set flag to True (change happened)

                
                if user_reward <= 0 and user.get_alligiance() == server:     # if there is no benefit the user goes unmatched
                    user.get_alligiance().remove_from_coalition(user)
                    user.change_server(None)
                    flag = True

    users.sort(key=lambda x: x.num)     # sort users by number
    servers.sort(key=lambda x: x.num)     # sort servers by number