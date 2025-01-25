from Classes.User import User
from Classes.Server import Server

# Parameters

alpha = 1
beta = 1
gamma = 1
delta = 1
epsilon = 1

constant = 1

w_local = 0.5
w_trans = 0.5

def calculate_weights(ratio):
    pow = 4

    if ratio == 1:
        w1 = 0.5
        w2 = 0.5
    elif ratio < 1:
        w2 = 0.5 * ratio**pow
        w1 = 0.5 * (2 - ratio**pow)
    else:
        w2 = 0.5 * (1 / ratio**(1./pow))
        w1 = 0.5 * (2 - 1 / ratio**(1./pow))

    return w1, w2

# User’s Utility function

def user_utility(user: User, server: Server, verbose = False):
    if server is None:
        return 0
    
    datarate = user.get_datarate(server)
    payment = user.get_payment(server)
    dataquality = user.get_dataquality(server)
    E_transmit = user.get_E_transmit(datarate)
    E_local = user.get_E_local()

    w_local, w_trans = calculate_weights(user.get_energy_ratio(server))

    utility = alpha * datarate + beta * payment*server.p - gamma * (w_local*E_local + w_trans*E_transmit) + delta * dataquality

    if verbose:
        print("Utility: ", utility)
        print("Datarate: ", datarate)
        print("Payment: ", payment*server.p)
        print("Energy Local: ", E_local)
        print("Energy Transmission: ", E_transmit)
        print("Dataquality: ", dataquality)
        print()

    return utility + constant    # + constant is to have positive utilities

# User Utility with Externality

def user_utility_ext(user: User, server: Server, external_denominator = None, external_denominator_payment = None, verbose = False):
    if server is None:
        return 0

    if external_denominator is None and external_denominator_payment is None:
        E_local, dataquality, payment, datarate, E_transmit = user.get_magnitudes(server)
    else:
        E_local, dataquality, payment, datarate, E_transmit = user.get_magnitudes(server, external_denominator=external_denominator, external_denominator_payment=external_denominator_payment)

    w_local, w_trans = calculate_weights(user.get_energy_ratio(server))

    utility = alpha * datarate + beta * payment*server.p - gamma * (w_local*E_local + w_trans*E_transmit) + delta * dataquality

    if verbose:
        print("Utility: ", utility)
        print("Datarate: ", datarate)
        print("Payment: ", payment*server.p)
        print("Energy Local: ", E_local)
        print("Energy Transmission: ", E_transmit)
        print("Dataquality: ", dataquality)
        print()

    return utility + constant    # + constant is to have positive utilities

#Server's Utility function

def server_utility(server, coalition, verbose = False):
    utility = 0
    
    for u in coalition:
        utility += user_utility(u,server,verbose=verbose) 

    utility -= epsilon * (server.p)**2
        
    return utility


#Server's Utility function with externality

def server_utility_externality(servers, coalition, server, verbose = False):
    utility = 0
    
    for u in coalition:
        utility += user_utility_ext(u,server,verbose=verbose) 

    utility -= epsilon * (server.p)**2

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