def check_matching_equality(servers, servers_list):
    for i, _servers in enumerate(servers_list):
        if matching_equality(servers, _servers):
            return i
    
    return None

def matching_equality(servers1, servers2):
    for s1,s2 in zip(servers1, servers2):

        s1_coalition = s1.get_coalition()
        s1_coalition = {u.num for u in s1_coalition}

        s2_coalition = s2.get_coalition()
        s2_coalition = {u.num for u in s2_coalition}

        if s1_coalition != s2_coalition:
            return False
        
    return True
        