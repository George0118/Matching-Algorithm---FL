# Server Class
class Server:
    def __init__(self, x, y, z, p, num):
        self._x = x
        self._y = y
        self._z = z
        self._p = p
        self._min_payment = 1
        self._Ns_max = p/self._min_payment
        self._num = num
        self.coalition = set()
        self.invitations = set()

    def get_invitation(self, user):         # Getting Invited by user
        self.invitations.add(user)

    def add_to_coalition(self, user):         # Add user to coalition
        self.coalition.add(user)

    def remove_from_coalition(self, user):    # Remove user from coalition
        self.coalition.remove(user)

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
    def p(self):
        return self._p
    
    @property
    def num(self):
        return self._num
    
    @property
    def Ns_max(self):
        return self._Ns_max
    
    def get_coalition(self):
        return self.coalition
    
    def get_invitations_list(self):
        return self.invitations
    
    def clear_invitations_list(self):
        self.invitations = set()