'''
@Title : Traditional Federated Learning
@Author : Zhilin Wang
@Email : wangzhil.edu
@Date : 3-08-2022
@Reference: https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399 
'''
from config import num
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(num)  # Use GPU device
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from Data.Classes.Model import Model
from Data.Classes.Client import Client
from collections import deque
import time
from config import N

def Servers_FL(users, servers, R, lr, epoch, X_users, y_users, X_server, y_server):

  gpus = tf.config.list_physical_devices('GPU')
  print(gpus)
  print()

  server_losses = []
  server_accuracy = []

  user_losses = [[] for i in range(N)]
  user_accuracies = [[] for i in range(N)]

  for i in tf.range(len(servers)):

    server = servers[i]
    print("Training model for Server ", server.num, "\n")

    baseModel = Model().base_model()

    class_weights = [None] * len(users)
    X_train = [None] * len(users)
    y_train = [None] * len(users)
    X_test = [None] * len(users)
    y_test = [None] * len(users)

    coalition = list(server.get_coalition())

    factors = server.factors_calculation()   # Calculate factors to multiply the weigths

    print(factors)
    
    # Specify labels for the specific server disaster
    y_server[i] = server.specify_disaster(y_server[i])  

    for j in tf.range(len(coalition)):
      u = coalition[j]
      y_users[u.num] = server.specify_disaster(y_users[u.num])
      X_train[u.num], X_test[u.num], y_train[u.num], y_test[u.num] = train_test_split(X_users[u.num], y_users[u.num], test_size=0.2, random_state=42)

      total_samples = 0
      class_0_samples = 0
      class_1_samples = 0  

      # Calculate class weights
      total_samples += len(y_train[u.num])
      class_0_samples += np.sum(y_train[u.num] == 0)
      class_1_samples += np.sum(y_train[u.num] == 1)

      if class_0_samples == 0 or class_1_samples == 0:
        class_0_weight = 1
        class_1_weight = 1
      else:
        class_0_weight = total_samples / (2 * class_0_samples)
        class_1_weight = total_samples / (2 * class_1_samples)

      class_weights[u.num] = {0: class_0_weight, 1: class_1_weight}


    X_train_server, X_test_server, y_train_server, y_test_server = train_test_split(X_server[i], y_server[i], test_size=0.5, random_state=42)

    # Feature Extraction for all
    print("Feature Extraction:")

    for j in tf.range(len(coalition)):
        u = coalition[j]
        X_train[u.num] = Model().extract_features(baseModel, X_train[u.num])
        X_test[u.num] = Model().extract_features(baseModel, X_test[u.num])
    
    X_train_server = Model().extract_features(baseModel, X_train_server)
    X_test_server = Model().extract_features(baseModel, X_test_server)

    print("Feature Extraction complete!\n")

    # Begin training
  
    global_model=Model().global_model(X_train_server.shape[1:])

    accuracy=[]
    losses=[]

    losses_history = deque(maxlen=3)
    accuracies_history = deque(maxlen=3)

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

        weix, loss, acc = client.training(X_train[u.num],
                              y_train[u.num],
                              global_weights,
                              X_train[u.num].shape[1:]
                              )
        
        val_loss,val_acc=client.evaluate(X_test[u.num], y_test[u.num], weix, X_test[u.num].shape[1:])
        
        user_losses[u.num].append(val_loss)
        user_accuracies[u.num].append(val_acc)

        weix = client.scale_model_weights(weix, factors[u.num])
        weit.append(weix)

      global_weight=server.sum_scaled_weights(weit) # fedavg
      print("Global Model:")
      global_model.set_weights(global_weight)
      Model().train_model(global_model, X_train_server, y_train_server)
      loss,acc=Model().evaluate_model(global_model, X_test_server, y_test_server)
      losses.append(loss)
      accuracy.append(acc)

      losses_history.append(loss)
      accuracies_history.append(acc)
      end_time = time.time()
      elapsed_time = end_time - start_time

      print(f"\nGlobal Round {r + 1} took {elapsed_time:.2f} seconds\n")

      # if len(losses_history) == 3 and max(losses_history) - min(losses_history) <= 0.05 and len(accuracies_history) == 3 and max(accuracies_history) - min(accuracies_history) <= 0.005 and r+1 >= 40:
      #   print("Stopping training.\n")
      #   break

    server_losses.append(losses)
    server_accuracy.append(accuracy)

    print("------------------------------------------------------------------")

  return server_losses, server_accuracy, user_losses, user_accuracies
