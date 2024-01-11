# Parameters

alpha = 1
beta = 1
gamma = 1
delta = 1
epsilon = 1

# User’s Utility function

def user_utility(user, server):
    index = server.num
    
    datarate_list = user.get_datarate()
    payment_list = user.get_payment()
    dataquality_list = user.get_dataquality()
    E_local = user.get_Elocal()
    
    utility = alpha * datarate_list[index] + beta * payment_list[index] - gamma * (E_local + 1/datarate_list[index]) + delta * dataquality_list[index]

    return utility

#Server's Utility function

def server_utility(server, users_list):
    utility = 0
    
    for i in range(len(users_list)):
        user = users_list[i]
        utility += user_utility(user,server) - epsilon * server.p^2
        
    return utility  

#Approximate Matching Function

def approximate_fedlearner_matching(unmatched_users, servers):
    
    server_availability = [True] * len(unmatched_users) # Initialize users' server availability
    flag = True # flag that shows at least one user has available servers
    
    while(unmatched_users and flag):

        for u in unmatched_users:       # Users sending invitations
            available_servers = u.get_available_servers()
            if(available_servers):      # If the user has available servers then it needs to find its favorite
                favorite_MEC = available_servers[0]
                max_utility = user_utility(u,favorite_MEC)
                for s in available_servers:
                    utility = user_utility(u,s)
                    if(utility > max_utility):
                        max_utility = utility
                        favorite_MEC = s

                favorite_MEC.get_invitation(u)

        for server in servers:          # Servers receiving invitations
            invitations = server.get_invitations_list()
            coalition = server.get_coalition()

            if(invitations):            # If the server has received invitations it needs to choose its favorite User to pair
                favorite_User = invitations[0]
                max_utility = server_utility(server, coalition + [invitations[0]])

                for u in invitations:
                    utility = server_utility(server, coalition + [u])
                    if(utility > max_utility):
                        max_utility = utility
                        favorite_User = u

                server.add_to_coalition(favorite_User)      # server adds favorite User to its Coalition 
                favorite_User.set_available_servers([])     # favorite User deletes all its available servers since he is matched

                for u in invitations:           # The rest of the users need to remove this server from their available servers
                    if(u != favorite_User):
                        available_servers = u.get_available_servers()
                        available_servers.remove(server)