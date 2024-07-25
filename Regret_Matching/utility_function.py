from Classes.Server import Server
from Classes.User import User, fn_max, ds_max

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

def user_utility(user: User, server: Server, verbose = False):
    if server is None:
        return 0
    
    index = server.num
    
    datarate = user.get_datarate(server)
    payment = user.get_payment(server)
    dataquality = user.get_dataquality(server)
    E_transmit = user.get_E_transmit(datarate)
    E_local = user.get_E_local()

    utility_functions = user.util_fun[index]
    
    utility =   utility_functions[0](datarate, E_transmit) + \
                utility_functions[1](payment, E_local*(ds_max/user.used_datasize)) + \
                utility_functions[2](dataquality, E_local*(fn_max/user.current_fn)**2)
    if verbose:
        print("Utility: ", utility)
        print("Datarate: ", datarate)
        print("Payment: ", payment)
        print("Energy Local: ", E_local)
        print("Energy Transmission: ", E_transmit)
        print("Dataquality: ", dataquality)
        print()

    return utility

# User Utility with Externality

def user_utility_ext(user: User, server: Server, external_denominator = None, verbose = False):
    if server is None:
        return 0
    
    index = server.num
    
    if external_denominator is None:
        E_local, dataquality, payment, datarate, E_transmit = user.get_magnitudes(server)
    else:
        E_local, dataquality, payment, datarate, E_transmit = user.get_magnitudes(server, external_denominator=external_denominator)

    utility_functions = user.util_fun[index]
    
    utility =   utility_functions[0](datarate, E_transmit) + \
                utility_functions[1](payment, E_local*(ds_max/user.used_datasize)) + \
                utility_functions[2](dataquality, E_local*(fn_max/user.current_fn)**2)
    
    if verbose:
        print("Utility: ", utility)
        print("Datarate: ", datarate)
        print("Payment: ", payment)
        print("Energy Local: ", E_local)
        print("Energy Transmission: ", E_transmit)
        print("Dataquality: ", dataquality)
        print("Util1: " , utility_functions[0](datarate, E_transmit))
        print("Util2: " , utility_functions[1](payment, E_local*(ds_max/user.used_datasize)))
        print("Util3: " , utility_functions[2](dataquality, E_local*(fn_max/user.current_fn)**2))
        print()

    return utility

