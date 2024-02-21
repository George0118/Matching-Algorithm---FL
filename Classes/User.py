import random

a = 2 * 10**(-28)
fn = 1 * 10**9
qn = 20

# User Class
class User:
    def __init__(self, x, y, z, num):
        self._x = x
        self._y = y
        self._z = z
        self._num = num
        self.datasize = 0
        self.E_local = (a * random.uniform(0.95, 1.05) * qn * random.uniform(0.95, 1.05) * self.datasize * fn**2)/2  # Energy Consumption to train model locally
        self.E_transmit = []
        self.E_transmit_ext = []    # Transmission Energy with externality
        self.belongs_to = None
        self.importance = []
        self.datarate = []
        self.datarate_ext = []  # Datarate with externality
        self.payment = []
        self.dataquality = []
        self.available_servers = []

    def set_datasize(self,value):       # Local Datasize of User
        self.datasize = value
        self.E_local = (a * random.uniform(0.95, 1.05) * qn * random.uniform(0.95, 1.05) * self.datasize * fn**2)/2 # Update E_local

    def add_importance(self, value):     # Normalized Data importance of user for each server
        self.importance.append(value)
        
    def add_datarate(self, value):       # Normalized Data rate of user for each server
        self.datarate.append(value)

    def add_datarate_ext(self, value):       # Normalized Data rate with externality of user for each server
        self.datarate_ext.append(value)
        
    def add_payment(self, value):        # Normalized Payment user will receive from each server
        self.payment.append(value)

    def add_dataquality(self, value):    # Normalized dataquality of user for each server
        self.dataquality.append(value)    

    def set_Elocal(self, value):    # Set user's normalized E_local
        self.E_local = value   

    def add_Etransmit(self, value):    # Normalized Energy Transmission
        self.E_transmit.append(value)   

    def add_Etransmit_ext(self, value):    # Normalized Energy Transmission
        self.E_transmit_ext.append(value)  
        
    def set_available_servers(self, list):  # Available server list for this user
        self.available_servers = list[:] 

    def change_server(self, server):
        self.belongs_to = server
        
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
    
    def get_datasize(self):
        return self.datasize
    
    def get_importance(self):
        return self.importance

    def get_datarate(self):
        return self.datarate
    
    def get_datarate_ext(self):
        return self.datarate_ext

    def get_payment(self):
        return self.payment
    
    def get_dataquality(self):
        return self.dataquality
    
    def get_Elocal(self):
        return self.E_local
    
    def get_Etransmit(self):
        return self.E_transmit
    
    def get_Etransmit_ext(self):
        return self.E_transmit_ext
    
    def get_available_servers(self):
        return self.available_servers
    
    def get_alligiance(self):
        return self.belongs_to
