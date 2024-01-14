# Critical Point Class
class CP:
    def __init__(self, x, y, z, num, disaster):
        self._x = x
        self._y = y
        self._z = z
        self._num = num
        self.disaster = disaster

    def config_disaster(self, value):
        self.disaster = value

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
    
    def get_disaster(self):
        return self.disaster