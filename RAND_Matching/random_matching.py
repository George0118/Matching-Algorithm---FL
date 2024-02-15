from typing import List
from Classes.User import User
from Classes.Server import Server
from general_parameters import *
import random

def random_fedlearner_matching(users: List[User], servers: List[Server]):
    
    matched = 0
    matched_users = [False]*N

    while(matched < N):
        random.shuffle(servers)     # shuffle servers to get rid of prioritization
        for s in servers:
            random_number = random.uniform(0, 1)

            random_user = random.choice(users)
            while(matched_users[random_user.num] == True):
                random_user = random.choice(users)
            
            if random_number > 0.99:    # User stays unmatched
                matched_users[random_user.num] = True
                matched += 1
            else:
                if len(s.get_coalition()) < s.Ns_max:   # Else if there is space in the server 
                    s.add_to_coalition(random_user)     # Add the random user to its coalition
                    random_user.change_server(s)
                    matched_users[random_user.num] = True
                    matched += 1

            if matched >= N:
                break

    servers.sort(key=lambda x: x.num)     # sort servers by number

                
