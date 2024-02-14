# Class that defines available actions for a user

class Action:
    def __init__(self, target, other_user = None):
        self.target = target
        self.exchange_user = other_user

