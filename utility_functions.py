# Parameters

alpha = 1
beta = 1
gamma = 1
delta = 1
epsilon = 1

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
    
    utility = alpha * datarate_list[index] + beta * payment_list[index]/(len(current_coalition)+1) - gamma * (E_local + E_transmit_list[index]) + delta * avg_dataquality

    # print("Utility: ", utility)
    # print("Datarate: ", datarate_list[index])
    # print("Payment: ", payment_list[index]/(len(current_coalition)+1))
    # print("Energy: ", E_local + E_transmit_list[index])
    # print("Dataquality: ", avg_dataquality)

    return utility

#Server's Utility function

def server_utility(server, coalition):
    utility = 0
    
    for u in coalition:
        utility += user_utility(u,server) 

    utility -= epsilon * server.p**2
        
    return utility  


#Server's Utility function with externality

def server_utility_externality(servers, coalition, server):
    utility = 0
    
    for u in coalition:
        utility += user_utility(u,server) 

    utility -= epsilon * server.p**2

    added_payment_of_rest = 0

    for s in servers:
        if(s != server):
            added_payment_of_rest += s.p

    utility = utility/added_payment_of_rest
        
    return utility  