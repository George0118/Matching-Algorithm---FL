from typing import List
from Classes.Server import Server
import random
import math
import numpy as np
import copy
from general_parameters import *

a = 2 * 10**(-28)
q = 20
fn_max = 2 * 10**9  # Cycles per Sec
Ptrans_max = 2  # Watt
ds_max = 500 * 3*224*224*8  # bits
E_local_max = a*1.05 * q*1.05 * ds_max * (fn_max)**2 # J
datarate_max = B    # bits/sec
E_transmit_max = Z*Ptrans_max*10/B  # J

quantized_datasize = [0.5, 0.6667, 0.8333, 1]
quantized_fn = [i * fn_max for i in [0.5, 0.6667, 0.8333, 1]]
quantized_ptrans = [i * Ptrans_max for i in [0.5, 0.6667, 0.8333, 1]]

# User Class
class User:
    def __init__(self, x, y, z, num):
        # Randomize each user's device
        self.an = a * random.uniform(0.95, 1.05)
        self.qn = q * random.uniform(0.95, 1.05)

        self._x = x
        self._y = y
        self._z = z
        self._num = num
        self.datasize = 0
        self.distances = []
        self.importance = []
        
        self.max_payment = []
        self.payments = []

        self.dataquality = []

        self.belongs_to = None

        # Variables
        self.current_fn = None
        self.current_ptrans = None
        self.used_datasize = None

        self.E_local_val = None
        self.E_transmit_ext_list = None    # List of Transmission Energies with externality
        self.datarate_ext_list = None  # List of Datarates with externality


    def set_distances(self, list):      # User distances from Servers
        self.distances = copy.deepcopy(list)

    def set_datasize(self,value):       # Local Datasize of User
        self.datasize = value
    
    def add_importance(self, value):     # Normalized Data importance of user for each server
        self.importance.append(value)

    def add_dataquality(self, value):    # Normalized dataquality of user for each server
        self.dataquality.append(value) 

    def change_server(self, server):
        self.belongs_to = server

    def set_energy_ratio(self, ratio):
        self.energy_ratio = ratio

    def set_parameters(self, fn, ptrans, datasize):
        self.current_fn = fn
        self.current_ptrans = ptrans
        self.used_datasize = datasize * self.datasize

    # Update magnitudes for all servers or for only the target server depending on need
    def set_magnitudes(self, servers, change_all=True, server_num = None, external_denominator = None):
        if change_all:
            self.set_E_local(self.E_local()/E_local_max)
            self.set_dataqualities([self.used_datasize * imp / ds_max for imp in self.importance])
            self.set_payments([self.current_fn * imp / fn_max for imp in self.importance])
            self.set_datarate([i/datarate_max for i in self.datarate_ext(servers)])
            self.set_E_transmit([i/E_transmit_max for i in self.E_transmit_ext()])
        else:
            self.set_E_local(self.E_local()/E_local_max)
            self.update_dataquality(server_num)
            self.update_payment(server_num)
            self.update_datarate(servers[server_num], external_denominator)
            self.update_E_transmit(server_num)

    def set_E_local(self, value):
        self.E_local_val = value

    def set_payments(self, list):
        self.payments = copy.deepcopy(list)

    def set_dataqualities(self, list):
        self.dataquality = copy.deepcopy(list)

    def set_E_transmit(self, list):
        self.E_transmit_ext_list = copy.deepcopy(list)
        
    def set_datarate(self, list):
        self.datarate_ext_list = copy.deepcopy(list)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z
    
    @property
    def num(self):
        return self._num
    
    # Calculate external influence of other users for each server
    def get_external_denominators(self, servers):
        external_denominators = []
        for server in servers:
            denominator_sum = 0
            user_group = set(server.get_coalition())

            # We need only external, so remove myself
            user_to_remove = next((user for user in user_group if user.num == self.num), None)
            if user_to_remove is not None:
                user_group.remove(user_to_remove)

            for u in user_group:
                g_ext = channel_gain(u.distances[server.num], u.num, server.num)
                denominator_sum += g_ext * u.current_ptrans * u.distances[server.num]
                
            external_denominators.append(denominator_sum)

        return external_denominators
    
    def get_datasize(self):
        return self.datasize
    
    def get_importance(self):
        return self.importance
    
    def get_payments(self):
        return self.payments
    
    def get_dataquality(self):
        return self.dataquality
    
    def get_alligiance(self):
        return self.belongs_to
    
    def get_E_local(self):
        return self.E_local_val
    
    def get_E_transmit_ext(self):
        return self.E_transmit_ext_list
    
    def get_datarate_ext(self):
        return self.datarate_ext_list
    
    # Magnitude Functions

    def E_local(self):      # E_local calculation
        return self.an * self.qn * self.used_datasize * self.current_fn**2
    
    def datarate_ext(self, servers: List[Server]):    # Datarate with extrenality calculation
        datarate_ext_list = []
        for server in servers:
            denominator_sum = 0
            user_group = set(server.get_coalition())
            user_group.add(self)
            for u in user_group:
                g_ext = channel_gain(u.distances[server.num], u.num, server.num)
                denominator_sum += g_ext * u.current_ptrans * u.distances[server.num]
                
            g = channel_gain(self.distances[server.num], self.num, server.num)
            dr = B*math.log2(1 + g*self.current_ptrans/(denominator_sum + I0))
            datarate_ext_list.append(dr)
        return datarate_ext_list 

    def E_transmit_ext(self):   # E_transmit with externality calculation
        E_transmit_ext_list = [Z*self.current_ptrans/(dr*datarate_max) for dr in self.datarate_ext_list]
        return E_transmit_ext_list
    
    # Update Functions

    def update_dataquality(self, server_num):      # Updating SINGLE payment
        self.dataquality[server_num] = self.used_datasize * self.importance[server_num] / ds_max

    def update_payment(self, server_num):      # Updating SINGLE payment
        self.payments[server_num] = self.current_fn * self.importance[server_num] / fn_max
    
    def update_datarate(self, server: Server, external_denominator):    # Updating SINGLE datarate
        denominator_sum = 0
        g_ext = channel_gain(self.distances[server.num], self.num, server.num)
        denominator_sum += g_ext * self.current_ptrans * self.distances[server.num]
        denominator_sum += external_denominator
            
        g = channel_gain(self.distances[server.num], self.num, server.num)
        dr = B*math.log2(1 + g*self.current_ptrans/(denominator_sum + I0))

        self.datarate_ext_list[server.num] = dr/datarate_max

    def update_E_transmit(self, server_num):   # Updating SINGLE Etransmit
        self.E_transmit_ext_list[server_num] = (Z*self.current_ptrans/(self.datarate_ext_list[server_num]*datarate_max))/E_transmit_max 



def channel_gain(distance, num_user, num_server):   # Channel gain calculation
    g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[num_server][num_user]

    g = 10**(-g / 10)

    return g
