import tensorflow as tf
import numpy as np
from Data.Classes.Model import Model

def importance_map(i):
    return 0.3 + 0.7*(1-i)

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
        self.critical_points = []
        self.coalition = set()
        self.invitations = set()

    def set_p(self, value):         # Set payment
        self._p = value

    def get_invitation(self, user):         # Getting Invited by user
        self.invitations.add(user)

    def add_to_coalition(self, user):         # Add user to coalition
        self.coalition.add(user)

    def remove_from_coalition(self, user):    # Remove user from coalition
        self.coalition.remove(user)

    def add_critical_point(self, cp):
        self.critical_points.append(cp)

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

    def get_critical_points(self):
        return self.critical_points


    # ============= Federated Learning ============== #
    
    def scale_model_weights(self,weight,scalar,num):
        '''function for scaling a models weights'''
        weight_final = []
        wei=[]
        steps = 8

        for j in range(num):
            fa=scalar[j]
            scar=[fa for k in range(steps)]
            for i in range(steps):
                weight_final.append(scar[i]*weight[j][i])
                #weight_final.append(weight[i])
            wei.append(weight_final)
        return wei

    def sum_scaled_weights(self,scaled_weight_list):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_grad = list()
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad

    def evaluate(self,model,test_X, test_y):
        loss,acc=Model.evaluate_model(model,test_X, test_y)
        return loss,acc

    def specify_disaster(self, labels):
        disaster = ''
        if self.num == 0:
            disaster = 'fire'
        elif self.num == 1:
            disaster = 'flood'
        elif self.num == 2:
            disaster = 'earthquake'

        if disaster == '':
            return labels

        labels = np.where(labels == disaster, 1, 0)

        return labels
    
    def factors_calculation(self, N):
        coalition = self.get_coalition()
        critical_points = self.get_critical_points()

        factors = [0] * N

        Ns = len(list(self.get_coalition()))

        denominator = 0
        for u in coalition:
            importance_list = u.get_importance()
            for cp in critical_points:
                denominator += importance_map(importance_list[cp.num]) * u.get_datasize() * Ns

        for u in coalition:
            importance_list = u.get_importance()
            numerator = 0
            for cp in critical_points:
                numerator += importance_map(importance_list[cp.num]) * u.get_datasize()
            factors[u.num] = numerator/denominator

        return factors

