import random
random.seed(42)
from Classes.Server import Server
from Classes.User import User
from typing import List
from GT_Matching.accurate_matching_conditions import *
from GT_Matching.utility_functions import *
from config import N,S
import copy
from pprint import pprint
from itertools import product

#Accurate Matching Function

rewards = [[0] * S for _ in range(N)]

def accurate_fedlearner_matching(original_apprx_matched_users: List[User], original_servers: List[Server]):
    global rewards

    apprx_matched_users = copy.deepcopy(original_apprx_matched_users)
    servers = copy.deepcopy(original_servers)

    # Set everything to 0 in case the global array its already changed from before
    rewards = [[0] * (S) for _ in range(N)]

    t = 1
    while(t <= 1000*N):

        random_user = apprx_matched_users[random.randint(0,N-1)]   # Select random user

        if(random_user.get_alligiance() is None):     # If user does not belong to any coalition
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
                rewards[random_user.num][favorite_MEC.num] += server_reward(servers, random_user, favorite_MEC)
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

                    favorite_MEC.remove_from_coalition(exchange_user.num)   # remove exchange user from server
                    exchange_user.change_server(None)
                    favorite_MEC.add_to_coalition(random_user)          # add random user to server
                    random_user.change_server(favorite_MEC)

                    rewards[random_user.num][favorite_MEC.num] += server_reward(servers, random_user, favorite_MEC)
                    # print("User from None joins server")

        else:   # else the user belongs in a coalition already

            for server in servers:
                if server.num == random_user.get_alligiance().num:
                    current_server = server
                    break

            other_server = random.choice(servers)      # select another server randomly
            while(other_server.num == current_server.num):     # different than the current server 
                other_server = random.choice(servers)

            if(len(other_server.get_coalition()) < other_server.Ns_max):      # If the other server has space
                utility_diff = check_user_changes_servers(servers, random_user, current_server, other_server)   # check if giving away a user is better
                
                if(utility_diff > 0):   # if it is better
                    current_server.remove_from_coalition(random_user.num)   # remove from current server
                    other_server.add_to_coalition(random_user)          # add to other server
                    random_user.change_server(other_server)

                    rewards[random_user.num][other_server.num] += server_reward(servers, random_user, other_server)
                    # print("User changed servers")   
                
                else:       # Reward the user for staying at the coalition
                    rewards[random_user.num][current_server.num] += server_reward(servers, random_user, current_server)

            else:   # if the other server has no space
                other_coalition = other_server.get_coalition()

                other_user_num = random.choice(list(other_coalition)).num  # select a random user from its coalition

                for user in apprx_matched_users:
                    if user.num == other_user_num:
                        other_user = user
                        break

                utility_diff = check_user_exchange(servers, random_user, other_user, current_server, other_server)   # check if the servers exchanging users is beneficial

                if(utility_diff > 0):   # if it is better

                    current_server.remove_from_coalition(random_user.num)   # remove random user from current server
                    other_server.add_to_coalition(random_user)          # add random user to other server
                    random_user.change_server(other_server)

                    other_server.remove_from_coalition(other_user.num)      # remove other user from other server
                    current_server.add_to_coalition(other_user)         # add other user to current server
                    other_user.change_server(current_server)

                    rewards[random_user.num][other_server.num] += server_reward(servers, random_user, other_server)
                    rewards[other_user.num][current_server.num] += server_reward(servers, other_user, current_server)
                
                    # print("User exchange")

                else:       # Reward the users for staying at their coalition
                    rewards[random_user.num][current_server.num] += server_reward(servers, random_user, current_server)
                    rewards[other_user.num][other_server.num] += server_reward(servers, other_user, other_server)

        # Lastly check if removing the user from its current coalition would be beneficial
        if(random_user.get_alligiance() != None):
            current_server = servers[random_user.get_alligiance().num]

            utility_diff = check_user_leaves_server(servers, random_user, current_server)

            if(utility_diff > 0):       # If the server benefits from excluding the user
                rewards[random_user.num][current_server.num] += server_reward(servers, random_user, current_server)
                current_server.remove_from_coalition(random_user.num)   # remove from coalition
                random_user.change_server(None)
                # print("User removed from server")

        count = 0
        for s in servers:
            count += len(list(s.get_coalition()))

        if count > N:
            print("Something went wrong...")
            break

        t += 1

    print("\nRewards:")
    pprint(rewards)
    print()

    final_matching(original_apprx_matched_users, original_servers)


def final_matching(users: List[User], servers: List[Server]):
    flag = True     # flag that checks change - if no change the end

    cartesian_product = list(product(servers, users))
    
    while(flag):
        flag = False
        random.shuffle(cartesian_product)     # shuffle cartesian product each iteration to not prioritize any server, user
        for server, user in cartesian_product:
            user_reward = rewards[user.num][server.num]

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
                if user_reward > rewards[user.num][user.get_alligiance().num]:  # if this server and user benefit more
                    current_server.remove_from_coalition(user.num)   # remove from its original server and
                    user.change_server(server)      # add the user to the server
                    server.add_to_coalition(user)
                    flag = True         # and set flag to True (change happened)
                    # print("Changed and server had space")

            # else if the user belongs to None and there is no space
            elif current_server is None and len(server.get_coalition()) == server.Ns_max:
                u_min_contribute = None
                for u in server.get_coalition():    # check for the minimum contributing user in the current coalition if the server can benefit from exchange
                    if u_min_contribute is None or rewards[u_min_contribute.num][server.num] > rewards[u.num][server.num]:
                        for u1 in users:
                            if u1.num == u.num:
                                u_min_contribute = u1
                                break

                if user_reward > rewards[u_min_contribute.num][server.num]:        # if it can
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
                    if u_min_contribute is None or rewards[u_min_contribute.num][server.num] > rewards[u.num][server.num]:
                        for u1 in users:
                            if u1.num == u.num:
                                u_min_contribute = u1
                                break

                # if it can and user wants
                if user_reward > rewards[u_min_contribute.num][server.num] and user_reward > rewards[user.num][user.get_alligiance().num]:        
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