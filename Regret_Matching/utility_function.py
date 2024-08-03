from Classes.Server import Server
from Classes.User import User, fn_max, ds_max, Ptrans_max

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
    
    payment = user.get_payment(server)
    E_local = user.get_E_local()
    datarate, E_transmit = user.get_datarate_Etransmit(server)
        
    utility_functions = user.util_fun[index]
    
    utility =   utility_functions[0](datarate, E_transmit) + \
                utility_functions[1](payment*ds_max/user.used_datasize, E_local*(ds_max/user.used_datasize)) + \
                utility_functions[2](payment*fn_max/user.current_fn, E_local*(fn_max/user.current_fn)**2)
    if verbose:
        print("Utility: ", utility)
        print("Datarate: ", datarate)
        print("Payment: ", payment)
        print("Energy Local: ", E_local)
        print("Energy Transmission: ", E_transmit)
        print()

    return utility

# User Utility with Externality

def user_utility_ext(user: User, server: Server, external_denominator = None, total_payments = None, verbose = False):
    if server is None:
        return 0
    
    index = server.num
    
    if external_denominator is None:
        E_local, payment_fn, payment_dn, datarate, E_transmit = user.get_magnitudes(server, verbose=verbose)
    else:
        E_local, payment_fn, payment_dn, datarate, E_transmit = user.get_magnitudes(server, external_denominator=external_denominator, verbose=verbose)

    utility_functions = user.util_fun[index]

    utility =   utility_functions[0](datarate, E_transmit) + \
                utility_functions[1](payment_fn, E_local*(ds_max/user.used_datasize)) + \
                utility_functions[2](payment_dn, E_local*(fn_max/user.current_fn)**2)
    
    if verbose:
        print("Utility: ", utility)
        print("Datarate: ", datarate)
        print("Payment Fn: ", payment_fn)
        print("Payment Dn: ", payment_dn)
        print("Energy Local Fn: ", E_local*(ds_max/user.used_datasize))
        print("Energy Local Dn: ", E_local*(fn_max/user.current_fn)**2)
        print("Energy Transmission: ", E_transmit)
        print("Util1: " , utility_functions[0](datarate, E_transmit))
        print("Util2: " , utility_functions[1](payment_fn, E_local*(ds_max/user.used_datasize)))
        print("Util3: " , utility_functions[2](payment_dn, E_local*(fn_max/user.current_fn)**2))
        print()

    return utility

