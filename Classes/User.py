from typing import List
from Classes.Server import Server
import random
import math
import numpy as np
import copy
from general_parameters import *
from helping_functions import channel_gain

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

        # Utility function Parameters

        self.util_fun = [[None, None, None] for _ in range(S)] # One for each server

        # ========================= #

        self._x = x
        self._y = y
        self._z = z
        self._num = num
        self.datasize = 0
        self.distances = []
        self.importance = []
        self.belongs_to = None

        # Variables
        self.current_fn = None
        self.current_ptrans = None
        self.used_datasize = None

    def set_distances(self, list):      # User distances from Servers
        self.distances = copy.deepcopy(list)

    def set_datasize(self,value):       # Local Datasize of User
        self.datasize = value
    
    def add_importance(self, value):     # Normalized Data importance of user for each server
        self.importance.append(value)

    def change_server(self, server):
        self.belongs_to = server

    def set_parameters(self, fn, ptrans, datasize):
        self.current_fn = fn
        self.current_ptrans = ptrans
        self.used_datasize = datasize * self.datasize

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
    
    # Get magnitudes for the target server 
    def get_magnitudes(self, server, external_denominator = None):

        E_local = self.E_local()/E_local_max
        dataquality = self.used_datasize * self.importance[server.num] / ds_max
        payment = self.current_fn * self.importance[server.num] / fn_max
        datarate, E_transmit = self.get_external_values(server, external_denominator)

        return E_local, dataquality, payment, datarate, E_transmit
    
    # Get only values that are influenced by external factors (other users)
    def get_external_values(self, server: Server, external_denominator = None):
        denominator_sum = 0
        user_group = set(server.get_coalition())

        if external_denominator is None:
            # We need only external, so remove myself
            user_to_remove = next((user for user in user_group if user.num == self.num), None)
            if user_to_remove is not None:
                user_group.remove(user_to_remove)

            for u in user_group:
                g_ext = channel_gain(u.distances[server.num], u.num, server.num)
                denominator_sum += g_ext * u.current_ptrans * u.distances[server.num]
        else:
            denominator_sum = external_denominator

        # Include myself to calculate the datarate
        g = channel_gain(self.distances[server.num], self.num, server.num)
        denominator_sum += g * self.current_ptrans * self.distances[server.num]

        datarate = B*math.log2(1 + g*self.current_ptrans/(denominator_sum + I0))/datarate_max
        E_transmit = (Z*self.current_ptrans/(datarate*datarate_max))/E_transmit_max

        return datarate, E_transmit

    def get_datasize(self):
        return self.datasize
    
    def get_importance(self):
        return self.importance
    
    def get_payment(self, server: Server):
        return self.current_fn * self.importance[server.num] / fn_max
    
    def get_dataquality(self, server: Server):
        return self.used_datasize * self.importance[server.num] / ds_max
    
    def get_alligiance(self):
        return self.belongs_to
    
    def get_E_local(self):
        return self.E_local() / E_local_max
    
    def get_E_transmit(self, datarate):
        return (Z*self.current_ptrans/(datarate*datarate_max))/E_transmit_max
    
    def get_datarate(self, server: Server):
        g = channel_gain(self.distances[server.num], self.num, server.num)
        datarate = B*math.log2(1 + g*self.current_ptrans/(g*self.current_ptrans+I0))/datarate_max
        return datarate

    # Magnitude Functions

    def E_local(self):      # E_local calculation
        return self.an * self.qn * self.used_datasize * self.current_fn**2
    
    def datarate_ext(self, servers: List[Server]):    # Datarate with extrenality calculation
        datarate_ext_list = []
        for server in servers:
            denominator_sum = 0
            for u in server.get_coalition().union({self}):
                g_ext = channel_gain(u.distances[server.num], u.num, server.num)
                denominator_sum += g_ext * u.current_ptrans * u.distances[server.num]
                
            g = channel_gain(self.distances[server.num], self.num, server.num)
            dr = B*math.log2(1 + g*self.current_ptrans/(denominator_sum + I0))
            datarate_ext_list.append(dr)
        return datarate_ext_list 

    def E_transmit_ext(self):   # E_transmit with externality calculation
        E_transmit_ext_list = [Z*self.current_ptrans/(dr*datarate_max) for dr in self.datarate_ext_list]
        return E_transmit_ext_list

    def set_available_servers(self, list):  # Available server list for this user
        self.available_servers = list[:] 

    def get_available_servers(self):
        return self.available_servers
