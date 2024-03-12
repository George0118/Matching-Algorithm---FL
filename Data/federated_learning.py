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

from Data.Classes.Model import Model
from Data.Classes.Client import Client
from collections import deque
import time

def Servers_FL(users, servers, R, lr, epoch, X_train, y_train, X_test, y_test):

  gpus = tf.config.list_physical_devices('GPU')
  print(gpus)
  print()

  server_losses = []
  server_accuracy = []

  for i in tf.range(len(servers)):

    server = servers[i]
    print("Training model for Server ", server.num, "\n")

    baseModel = Model().base_model()

    user_features = [None] * len(users)
    class_weights = [None] * len(users)

    X_test_server = None
    y_test_server = None

    coalition = list(server.get_coalition())

    # Concatenate all the testing data for the specific server
    for j in tf.range(len(coalition)):
      u = coalition[j]
      if(X_test_server is None):
        X_test_server = X_test[u.num]
        y_test_server = y_test[u.num]
      else:
        X_test_server = np.concatenate((X_test_server, X_test[u.num]))
        y_test_server = np.concatenate((y_test_server, y_test[u.num]))

    factors = server.factors_calculation()   # Calculate factors to multiply the weigths
    
    # Specify labels for the specific server disaster
    y_test_server = server.specify_disaster(y_test_server)  

    for j in tf.range(len(coalition)):
      u = coalition[j]
      y_train[u.num] = server.specify_disaster(y_train[u.num])

      total_samples = 0
      class_0_samples = 0
      class_1_samples = 0  

      # Calculate class weights
      total_samples += len(y_train[u.num])
      class_0_samples += np.sum(y_train[u.num] == 0)
      class_1_samples += np.sum(y_train[u.num] == 1)

      class_0_weight = total_samples / (2 * class_0_samples)
      class_1_weight = total_samples / (2 * class_1_samples)

      class_weights[u.num] = {0: class_0_weight, 1: class_1_weight}


    # Feature Extraction for all
    print("Feature Extraction:")

    for j in tf.range(len(coalition)):
        u = coalition[j]
        user_features[u.num] = Model().extract_features(baseModel, X_train[u.num])
    
    server_features = Model().extract_features(baseModel, X_test_server)

    # Begin training
  
    global_model=Model().global_model(server_features.shape[1:])

    accuracy=[]
    losses=[]

    accuracy_history = deque(maxlen=3)

    for r in tf.range(R):
      print("------------------------------------------------------------------")
      print(f"Round: {r + 1}\n")
      start_time = time.time()

      global_weights=global_model.get_weights()
      weit=[]

      for j in tf.range(len(coalition)):
        u = coalition[j]
        print("User ", u.num, ":")
        client = Client(lr, epoch, u.num)

        weix = client.training(user_features[u.num],
                              y_train[u.num],
                              global_weights,
                              class_weights[u.num],
                              user_features[u.num].shape[1:]
                              )
        weix = client.scale_model_weights(weix, factors[u.num])
        weit.append(weix)

      global_weight=server.sum_scaled_weights(weit) # fedavg
      print("Global Model:")
      global_model.set_weights(global_weight)
      loss,acc=Model().evaluate_model(global_model, server_features, y_test_server)
      losses.append(loss)
      accuracy.append(acc)

      accuracy_history.append(acc)
      end_time = time.time()
      elapsed_time = end_time - start_time

      print(f"\nGlobal Round {r + 1} took {elapsed_time:.2f} seconds\n")

      # if len(accuracy_history) == 3 and max(accuracy_history) - min(accuracy_history) <= 0.005 and r+1 >= 10:
      #   print("Stopping training. Three consecutive accuracy differences are within 0.005.\n")
      #   break

    server_losses.append(losses)
    server_accuracy.append(accuracy)

    print("------------------------------------------------------------------")

  return server_losses, server_accuracy

