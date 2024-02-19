from GT_Matching.utility_functions import server_utility_externality

# Free User joins Server - TC1

def check_user_joins_server(servers, user, server):
    coalition = server.get_coalition()     # get current coalition of server 

    # Get difference in utility 
    diff = server_utility_externality(servers, coalition.union({user}), server) \
          - server_utility_externality(servers, coalition, server)

    return diff

# User leaves Server - TC2
    
def check_user_leaves_server(servers, user, server):
    coalition = server.get_coalition()     # get current coalition of server 

    # Get difference in utility 
    diff = server_utility_externality(servers, coalition.difference({user}), server)\
          - server_utility_externality(servers, coalition ,server)
    
    return diff

# User leaves Server to join another - TC3
    
def check_user_changes_servers(servers, user, server1, server2):
    coalition1 = server1.get_coalition()    # get coalition of server 1
    coalition2 = server2.get_coalition()    # get coalition of server 2

    # Get difference in utility
    diff = server_utility_externality(servers, coalition1.difference({user}), server1)\
          + server_utility_externality(servers, coalition2.union({user}), server2)\
          - server_utility_externality(servers, coalition1, server1)\
          - server_utility_externality(servers, coalition2, server2)
    
    return diff


# Servers exchange two users - TC4
    
def check_user_exchange(servers, user1, user2, server1, server2):
    coalition1 = server1.get_coalition()    # get coalition of server 1
    coalition2 = server2.get_coalition()    # get coalition of server 2

    # Get difference in utility
    diff = server_utility_externality(servers, coalition1.difference({user1}).union({user2}), server1)\
          + server_utility_externality(servers, coalition2.difference({user2}).union({user1}), server2)\
          - server_utility_externality(servers, coalition1, server1)\
          - server_utility_externality(servers, coalition2, server2)
    
    return diff