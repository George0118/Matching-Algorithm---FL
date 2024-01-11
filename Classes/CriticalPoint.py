# Critical Point Class
class CP:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z
        self.disaster = ""

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