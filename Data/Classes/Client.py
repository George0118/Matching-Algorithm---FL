import tensorflow as tf
from Data.Classes.Model import Model
from keras.optimizers.legacy import SGD
import keras
from itertools import combinations
import numpy as np

class Client:
  
  def __init__(self,lr,epoch,u_num):
    self.u_num = u_num
    self.epoch=epoch
    self.lr = lr
    self.loss=keras.losses.BinaryCrossentropy(),
    self.metrics=[keras.metrics.BinaryAccuracy(name="acc")]
    self.optimizer = SGD(learning_rate=self.lr, 
                decay=self.lr / 2, 
                momentum=0.9
               )
    
  def weight_client(self,data,m,n):
    wei_client = []
    for i in range(n):
        len_data = len(data[i])
        proba = len_data / m
        wei_client.append(proba)
    return wei_client
    
  
  def scale_model_weights(self,weight,scalar,num):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)

    fac=scalar[num]

    sca=[fac for i in range(steps)]

    for i in range(steps):
      weight_final.append(sca[i]*weight[i])
      
    return weight_final

  def training(self, features, labels, global_weights, shape, n_splits = 3):
      model = Model().global_model(shape)
      model.compile(optimizer="rmsprop",
                    loss=self.loss,
                    metrics=self.metrics
                    )
      model.set_weights(global_weights)

      features_splits = np.array_split(features, n_splits, axis=0)
      labels_splits = np.array_split(labels, n_splits, axis=0)

      all_combinations = list(combinations(range(n_splits), n_splits - 1))
      
      cv_weights = []

      print("CrossValidation")

      for index_combination in all_combinations:

        concatenated_features = np.concatenate([features_splits[i] for i in index_combination], axis=0)
        concatenated_labels = np.concatenate([labels_splits[i] for i in index_combination], axis=0)  

        model.fit(concatenated_features, concatenated_labels, epochs=self.epoch)

        cv_weights.append(model.get_weights())

      avg_weights = [sum(weight[i] for weight in cv_weights) / n_splits for i in range(len(cv_weights[0]))]

      print()

      return avg_weights