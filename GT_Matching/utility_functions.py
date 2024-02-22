import math

# Parameters

alpha = 1
beta = 1
gamma = 1
delta = 1
epsilon = 1

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

def user_utility(user, server):
    index = server.num
    current_coalition = server.get_coalition()

    critical_points = server.get_critical_points()
    
    datarate_list = user.get_datarate()
    payment_list = user.get_payment()
    dataquality_list = user.get_dataquality()
    E_transmit_list = user.get_Etransmit()

    # Calculate average dataquality of user depending on the Critical Points that concern this server
    avg_dataquality = 0
    for cp in critical_points:
        avg_dataquality += dataquality_list[cp.num]

    avg_dataquality = avg_dataquality/len(critical_points)

    E_local = user.get_Elocal()

    w_local, w_trans = calculate_weights(user.get_energy_ratio())
    
    utility = alpha * datarate_list[index] + beta * payment_list[index]/(len(current_coalition)+1) - gamma * (w_local*E_local + w_trans*E_transmit_list[index]) + delta * avg_dataquality

    # print("Utility: ", utility)
    # print("Datarate: ", datarate_list[index])
    # print("Payment: ", payment_list[index]/(len(current_coalition)+1))
    # print("Energy: ", E_local + E_transmit_list[index])
    # print("Dataquality: ", avg_dataquality)
    # print()

    return utility

# User Utility with Externality

def user_utility_ext(user, server, verbose = False):
    index = server.num
    current_coalition = server.get_coalition()

    critical_points = server.get_critical_points()
    
    datarate_list = user.get_datarate_ext()     # Get datarate with externality
    payment_list = user.get_payment()
    dataquality_list = user.get_dataquality()
    E_transmit_list = user.get_Etransmit_ext()  # Get Etransmit with externality

    # Calculate average dataquality of user depending on the Critical Points that concern this server
    avg_dataquality = 0
    for cp in critical_points:
        avg_dataquality += dataquality_list[cp.num]

    avg_dataquality = avg_dataquality/len(critical_points)

    E_local = user.get_Elocal()

    w_local, w_trans = calculate_weights(user.get_energy_ratio())

    # print("W_local:", w_local, "| W_Trans:", w_trans)
    
    utility = alpha * datarate_list[index] + beta * payment_list[index]/(len(current_coalition)+1) - gamma * (w_local*E_local + w_trans*E_transmit_list[index]) + delta * avg_dataquality

    if verbose:
        print("Utility: ", utility)
        print("Datarate: ", datarate_list[index])
        print("Payment: ", payment_list[index]/(len(current_coalition)+1))
        print("Energy Local: ", E_local)
        print("Energy Transmission: ", E_transmit_list[index])
        print("Dataquality: ", avg_dataquality)
        print()

    return utility

#Server's Utility function

def server_utility(server, coalition):
    utility = 0
    
    for u in coalition:
        utility += user_utility(u,server) 

    utility -= epsilon * math.sqrt(server.p)
        
    return utility  


#Server's Utility function with externality

def server_utility_externality(servers, coalition, server):
    utility = 0
    
    for u in coalition:
        utility += user_utility_ext(u,server) 

    utility -= epsilon * math.sqrt(server.p)

    added_payment_of_rest = 0

    for s in servers:
        if(s != server):
            added_payment_of_rest += s.p

    utility = utility/added_payment_of_rest

    # print("Utility: ", utility)
    # print(server.p)
        
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
    return server_utility_externality(servers, server.get_coalition(), server) \
            - server_utility_externality(servers, server.get_coalition().difference({user}), server)     

 
def user_reward(user, server):

    index = server.num
    current_coalition = server.get_coalition()

    critical_points = server.get_critical_points()
    
    datarate_list = user.get_datarate_ext()     # Get datarate with externality
    payment_list = user.get_payment()
    dataquality_list = user.get_dataquality()
    E_transmit_list = user.get_Etransmit_ext()  # Get Etransmit with externality

    # Calculate average dataquality of user depending on the Critical Points that concern this server
    avg_dataquality = 0
    for cp in critical_points:
        avg_dataquality += dataquality_list[cp.num]

    avg_dataquality = avg_dataquality/len(critical_points)

    E_local = user.get_Elocal()

    w_local, w_trans = calculate_weights(user.get_energy_ratio())

    # print("W_local:", w_local, "| W_Trans:", w_trans)
    
    reward = alpha * datarate_list[index] + beta * payment_list[index]/len(current_coalition) - gamma * (w_local*E_local + w_trans*E_transmit_list[index]) + delta * avg_dataquality

    return reward