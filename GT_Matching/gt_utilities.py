import math
from Regret_Matching.utility_function import *

# Server's Utility function

def server_utility(server, coalition, verbose = False):
    utility = 0
    
    for u in coalition:
        utility += user_utility(u,server,verbose=verbose) 

    utility -= epsilon * math.sqrt(server.p)
        
    return utility


# Server's Utility function with externality

def server_utility_externality(servers, coalition, server, verbose = False):
    utility = 0
    
    for u in coalition:
        utility += user_utility_ext(u,server,verbose=verbose) 

    utility -= epsilon * math.sqrt(server.p)

    added_payment_of_rest = 0

    for s in servers:
        if(s != server):
            added_payment_of_rest += s.p

    utility = utility/added_payment_of_rest
        
    return utility 

def user_utility_diff_exchange(user1, user2, server1, server2 = None):
    if server2 is not None:
        return user_utility_ext(user1, server2) + user_utility_ext(user2, server1) - user_utility_ext(user1, server1) - user_utility_ext(user2, server2)
    else:
        return user_utility_ext(user2, server1) - user_utility_ext(user1, server1) 


def user_utility_diff_servers(user, server1, server2):
    return user_utility_ext(user, server1) - user_utility_ext(user, server2)


# Reward Functions

def server_reward(servers, user, server):
    rew = server_utility_externality(servers, server.get_coalition(), server) \
            - server_utility_externality(servers, server.get_coalition().difference({user}), server)
    # if (rew < 0):
    #     print(rew)
    return rew    