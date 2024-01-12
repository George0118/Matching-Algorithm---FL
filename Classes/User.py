# User Class
class User:
    def __init__(self, x, y, z, num):
        self._x = x
        self._y = y
        self._z = z
        self._num = num
        self.E_local = 1        # Energy Consumption to train model locally: We assume its the same for all users
        self.belongs_to = None
        self.importance = []
        self.datarate = []
        self.payment = []
        self.dataquality = []
        self.available_servers = []

    def add_importance(self, value):     # Normalized Data importance of user for each server
        self.importance.append(value)
        
    def add_datarate(self, value):       # Normalized Data rate of user for each server
        self.datarate.append(value)
        
    def add_payment(self, value):        # Normalized Payment user will receive from each server
        self.payment.append(value)

    def add_dataquality(self, value):    # Normalized Payment user will receive from each server
        self.dataquality.append(value)    

    def set_Elocal(self, value):    # Set user's E_local
        self.E_local = value   
        
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
    
    def get_importance(self):
        return self.importance

    def get_datarate(self):
        return self.datarate

    def get_payment(self):
        return self.payment
    
    def get_dataquality(self):
        return self.dataquality
    
    def get_Elocal(self):
        return self.E_local
    
    def get_available_servers(self):
        return self.available_servers
    
    def get_alligiance(self):
        return self.belongs_to
