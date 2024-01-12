from utility_functions import user_utility, server_utility
import random

#Approximate Matching Function

def approximate_fedlearner_matching(unmatched_users, servers):
    
    server_availability = [True] * len(unmatched_users)         # Initialize users' server availability
    flag = True      # flag that shows at least one user has available servers
    
    while(flag):

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

            # If the server has received invitations and has space it needs to choose its favorite User to pair
            if(invitations and len(server.get_coalition()) + 1 <= server.Ns_max):   
                favorite_User = list(invitations)[0]
                max_utility = server_utility(server, coalition.union({favorite_User}))

                for u in invitations:
                    utility = server_utility(server, coalition.union({u}))
                    if(utility > max_utility):
                        max_utility = utility
                        favorite_User = u

                server.add_to_coalition(favorite_User)      # server adds favorite User to its Coalition 
                # print("Favorite User = ", favorite_User.num)
                favorite_User.change_server(server)     # favorite User now belongs to server
                favorite_User.set_available_servers([])     # favorite User deletes all its available servers since he is matched

                # num_list = [obj.num for obj in invitations]
                # print("Invitations for server ", server.num," are ", num_list)

                for u in invitations:           # The rest of the users need to remove this server from their available servers
                    if(u != favorite_User):
                        # print("Removing server ", server.num," from user", u.num)
                        available_servers = u.get_available_servers()
                        # num_list = [obj.num for obj in available_servers]
                        # print("User ", u.num, "available servers are: ", num_list)
                        available_servers.remove(server)
                        u.set_available_servers(available_servers)          # Update user's new available servers

                server.clear_invitations_list()

        for u in unmatched_users:                   # if a user has not more available servers set its flag to False
            if(not u.get_available_servers()):
                server_availability[u.num] = False

        # print(server_availability)
        
        flag = False                                # if there are no available servers for any user set the global flag to False
        for f in server_availability:
            if(f):
                flag = True 
                break