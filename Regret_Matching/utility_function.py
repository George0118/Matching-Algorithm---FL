from Classes.Server import Server
from Classes.User import User, E_local_max, E_transmit_max

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


# User Utility with Externality

def user_utility_ext(user: User, server: Server, verbose = False):
    if server is None:
        return 0
    
    index = server.num
    current_coalition = server.get_coalition()

    critical_points = server.get_critical_points()
    
    datarate_list = user.get_datarate_ext()     # Get datarate with externality
    payment = user.get_payments()[index]
    dataquality_list = user.get_dataquality()
    E_transmit_list = user.get_E_transmit_ext()  # Get Etransmit with externality
    E_local = user.get_E_local()

    # Calculate average dataquality of user depending on the Critical Points that concern this server
    avg_dataquality = 0
    for cp in critical_points:
        avg_dataquality += dataquality_list[cp.num]

    avg_dataquality = avg_dataquality/len(critical_points)

    # Normalize payment for users in server
    total_payment = 0
    current_coalition.add(user)
    for u in current_coalition:
        total_payment += u.get_payments()[index]

    w_local, w_trans = calculate_weights(E_local*E_local_max/(E_transmit_list[index]*E_transmit_max))
    
    utility = alpha * datarate_list[index] + beta * payment/total_payment - gamma * (w_local*E_local + w_trans*E_transmit_list[index]) + delta * avg_dataquality

    if verbose:
        print("Utility: ", utility)
        print("Datarate: ", datarate_list[index])
        print("Payment: ", payment/total_payment)
        print("Energy Local: ", E_local)
        print("Energy Transmission: ", E_transmit_list[index])
        print("Dataquality: ", avg_dataquality)
        print()

    return utility + constant    # + constant is to have positive utilities

