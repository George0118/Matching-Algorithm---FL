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
from Data.FL_helping_functions import *
from tensorflow import data as tf_data
from collections import deque
import time
import threading

def Servers_FL(users, servers, K, lr, epoch):

  gpus = tf.config.list_physical_devices('GPU')
  print(gpus)
  print()

  get_data=Get_data(users, servers)

  X_train, y_train, X_test, y_test=get_data.pre_data()

  X_test_total = np.empty_like(X_test[0])
  y_test_total = np.empty_like(y_test[0])
  models = []

  server_losses = []
  server_accuracy = []

  for server in servers:

    print("Training model for Server ", server.num, "\n")

    baseModel = Model().base_model()

    user_features = [None] * len(users)
    class_weights = [None] * len(users)

    X_test_server = np.empty_like(X_test[0])
    y_test_server = np.empty_like(y_test[0])

    # Concatenate all the testing data for the specific server
    for u in server.get_coalition():
      X_test_server = np.concatenate((X_test_server, X_test[u.num]))
      y_test_server = np.concatenate((y_test_server, y_test[u.num]))

    X_test_total = np.concatenate((X_test_total, X_test_server))
    y_test_total = np.concatenate((y_test_total, y_test_server))

    factors = server.factors_calculation(len(users))   # Calculate factors to multiply the weigths
    
    # Specify labels for the specific server disaster
    y_test_server = server.specify_disaster(y_test_server)  

    total_samples = 0
    class_0_samples = 0
    class_1_samples = 0  

    for u in server.get_coalition():
      y_train[u.num] = server.specify_disaster(y_train[u.num])

      # Calculate class weights
      total_samples += len(y_train[u.num])
      class_0_samples += np.sum(y_train[u.num] == 0)
      class_1_samples += np.sum(y_train[u.num] == 1)

      class_0_weight = total_samples / (2 * class_0_samples)
      class_1_weight = total_samples / (2 * class_1_samples)

      class_weights[u.num] = {0: class_0_weight, 1: class_1_weight}


    # Feature Extraction for all
    print("Feature Extraction:")

    threads = []

    for u in server.get_coalition():
        thread = threading.Thread(target=extract_features_wrapper, args=(u.num, baseModel, X_train, user_features))
        threads.append(thread)
        thread.start()

    for thread in threads:
      thread.join()
    
    server_features = Model().extract_features(baseModel, X_test_server)
    print(server_features.shape)
    print()

    # Begin training
  
    global_model=Model().global_model(server_features.shape[1:])

    accuracy=[]
    losses=[]

    accuracy_history = deque(maxlen=3)

    for k in range(K):
      print("------------------------------------------------------------------")
      print("Round: ", k+1, "\n")
      start_time = time.time()

      global_weights=global_model.get_weights()
      weit=[]

      threads = []

      for u in server.get_coalition():
        thread = threading.Thread(target=training_wrapper, args=(lr, epoch, u.num, user_features, y_train, global_weights, class_weights, factors, weit))
        threads.append(thread)
        thread.start()

      for thread in threads:
        thread.join()

      global_weight=server.sum_scaled_weights(weit) # fedavg
      print("Global Model:")
      global_model.set_weights(global_weight)
      loss,acc=Model().evaluate_model(global_model,server_features, y_test_server)
      losses.append(loss)
      accuracy.append(acc)

      accuracy_history.append(acc)
      end_time = time.time()
      elapsed_time = end_time - start_time

      print(f"\nGlobal Round {k + 1} took {elapsed_time:.2f} seconds\n")

      if len(accuracy_history) == 3 and max(accuracy_history) - min(accuracy_history) <= 0.005 and k+1 >= 10:
        print("Stopping training. Three consecutive accuracy differences are within 0.005.\n")
        break

    server_losses.append(losses)
    server_accuracy.append(accuracy)
    models.append(global_model)

    print("------------------------------------------------------------------")

  return server_losses, server_accuracy, X_test_total, y_test_total, models

