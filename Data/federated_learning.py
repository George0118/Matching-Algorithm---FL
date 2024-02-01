'''
@Title : Traditional Federated Learning
@Author : Zhilin Wang
@Email : wangzhil.edu
@Date : 3-08-2022
@Reference: https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399 
'''
import tensorflow as tf
import numpy as np
tf.test.gpu_device_name()

from Data.Classes.Client import Client
from Data.Classes.Data import Get_data
from Data.Classes.Model import Model
from tensorflow import data as tf_data
from collections import deque


def Servers_FL(users, servers, K, lr, epoch):

  gpus = tf.config.list_physical_devices('GPU')
  print(gpus)
  print()
  if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14800)])

  strategy = tf.distribute.MirroredStrategy()

  get_data=Get_data(users, servers)

  X_train, y_train, X_test, y_test=get_data.pre_data()

  server_losses = []
  server_accuracy = []

  for server in servers:

    print("Training model for Server ", server.num, "\n")

    X_test_server = np.empty_like(X_test[0])
    y_test_server = np.empty_like(y_test[0])

    # Concatenate all the testing data for the specific server
    for u in server.get_coalition():
      X_test_server = np.concatenate((X_test_server, X_test[u.num]))
      y_test_server = np.concatenate((y_test_server, y_test[u.num]))

    y_test_server = server.specify_disaster(y_test_server)    # set labels for specific disaster

    for u in server.get_coalition():
      y_train[u.num] = server.specify_disaster(y_train[u.num])

    with strategy.scope():
      global_model=Model().global_model()

      accuracy=[]
      losses=[]

      factors = server.factors_calculation(len(users))   # Calculate factors to multiply the weigths

      accuracy_history = deque(maxlen=3)

      for k in range(K):
        print(k)
        global_weights=global_model.get_weights()
        weit=[]

        for u in server.get_coalition():
          client=Client(lr,epoch)

          weix=client.training(X_train[u.num],y_train[u.num],global_weights)
          weix=client.scale_model_weights(weix,factors,u.num)
          weit.append(weix)

        global_weight=server.sum_scaled_weights(weit) # fedavg
        global_model.set_weights(global_weight)
        loss,acc=Model().evaluate_model(global_model,X_test_server,y_test_server)
        losses.append(loss)
        accuracy.append(acc)

        accuracy_history.append(acc)
        if len(accuracy_history) == 3 and max(accuracy_history) - min(accuracy_history) <= 0.005 and k >= 10:
          print("Stopping training. Three consecutive accuracy differences are within 0.005.\n")
          break

    server_losses.append(losses)
    server_accuracy.append(accuracy)

  return server_losses, server_accuracy

