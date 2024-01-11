# Server Class
class Server:
    def __init__(self, x, y, z, p, num):
        self._x = x
        self._y = y
        self._z = z
        self._p = p
        self._num = num
        self.coalition = []
        self.invitations = []

    def get_invitation(self, user):         # Getting Invited by user
        self.invitations.append(user)

    def add_to_coalition(self, user):         # Add user to coalition
        self.coalition.append(user)

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
    
    def get_coalition(self):
        return self.coalition
    
    def get_invitations_list(self):
        return self.invitations
    
    def clear_invitations_list(self):
        self.invitations = []
        return 0