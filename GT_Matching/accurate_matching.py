import random
import math
from accurate_matching_conditions import *
from general_parameters import *

#Accurate Matching Function

def accurate_fedlearner_matching(apprx_matched_users, servers):
    flag = True     # Flag that shows whether any changes occured

    while(flag):
        flag = False    # Set flag to False and if change occurs then set it to True

        for i in range(round(math.log2(N)*N)):

            random_user = apprx_matched_users[random.randint(0,N-1)]   # Select random user

            if(random_user.get_alligiance() == None):     # If user does not belong to any coalition
                max_utility_diff = 0
                favorite_MEC = None
                for s in servers:       # Find the server that benefits more from him
                    utility_diff = check_user_joins_server(servers,random_user,s)
                    if(utility_diff > max_utility_diff):
                        max_utility_diff = utility_diff
                        favorite_MEC = s

                if(favorite_MEC != None):   # If there is a server that benefits from the user, add the user to its coalition
                    favorite_MEC.add_to_coalition(random_user)
                    random_user.change_server(favorite_MEC)
                    flag = True     # and set flag to True
                    print("Added user to server")

            else:   # else the user belongs in a coalition already

                for j in range(round(math.log2(S)*S)):

                    current_server = random_user.get_alligiance()

                    other_server = random.choice(servers)      # select another server randomly
                    while(other_server == current_server):     # different than the current server 
                        other_server = random.choice(servers)

                    if(len(other_server.get_coalition()) + 1 <= other_server.Ns_max):      # If the other server has space
                        utility_diff = check_user_changes_servers(servers, random_user, current_server, other_server)   # check if giving away a user is better
                        
                        if(utility_diff > 0):   # if it is better
                            current_server.remove_from_coalition(random_user)   # remove from current server
                            other_server.add_to_coalition(random_user)          # add to other server
                            random_user.change_server(other_server)  
                            flag = True     # and set flag to True 
                            print("User changed servers")         

                    else:   # if the other server has no space
                        other_coalition = other_server.get_coalition()
                        for k in range(round(math.log2(len(list(other_coalition)))*len(list(other_coalition)))):

                            other_user = random.choice(list(other_coalition))  # select a random user from its coalition

                            utility_diff = check_user_exchange(servers, random_user, other_user, current_server, other_server)   # check if the servers exchanging users is beneficial

                            if(utility_diff > 0):   # if it is better
                                current_server.remove_from_coalition(random_user)   # remove random user from current server
                                other_server.add_to_coalition(random_user)          # add random user to other server
                                random_user.change_server(other_server)

                                other_server.remove_from_coalition(other_user)      # remove other user from other server
                                current_server.add_to_coalition(other_user)         # add other user to current server
                                other_user.change_server(current_server)

                                flag = True     # and set flag to True
                                print("User exchange")
                                break

            # Lastly check if removing the user from its current coalition would be beneficial
            if(random_user.get_alligiance() != None):
                current_server = random_user.get_alligiance()

                diff = check_user_leaves_server(servers, random_user, current_server)

                if(diff > 0):       # If the server benefits from excluding the user
                    current_server.remove_from_coalition(random_user)   # remove from coalition
                    random_user.change_server(None)
                    flag = True     # and set flag to True
                    print("User removed from server")

