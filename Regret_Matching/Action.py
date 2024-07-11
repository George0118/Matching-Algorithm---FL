# Class that defines available actions for a user
from Classes.Server import Server

class Action:
    def __init__(self, target: Server, fn, ptrans, ds):
        self.target = target
        self.fn = fn
        self.ptrans = ptrans
        self.ds = ds

