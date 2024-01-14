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
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from tensorflow import data as tf_data

# ======== Data Augmentation ========= #

data_augmentation_layers = [
    RandomFlip("horizontal"),
    RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# ===================================== #

def Servers_FL(users, servers, K, lr, epoch):

  print(tf.config.list_physical_devices('GPU'))

  client=Client(lr,epoch)
  get_data=Get_data(len(users))

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

    global_model=Model().global_model()

    accuracy=[]
    losses=[]
    
    factors = server.factors_calculation(X_train[0].nbytes, len(users))   # Calculate factors to multiply the weigths

    for k in range(K):
      print(k)
      global_weights=global_model.get_weights()
      weit=[]

      for u in server.get_coalition():
        train_ds = tf.data.Dataset.from_tensor_slices((X_train[u.num], y_train[u.num]))   # get the dataset for user u
        
        train_ds = train_ds.map(    # Apply `data_augmentation` to the training images.
            lambda img, label: (data_augmentation(img), label),
            num_parallel_calls=tf_data.AUTOTUNE,
        )

        # Split to X and y again
        X_train_u = []
        y_train_u = []

        for img, label in train_ds.as_numpy_iterator():
            X_train_u.append(img)
            y_train_u.append(label)

        # Convert lists to NumPy arrays
        X_train_u = np.array(X_train_u)
        y_train_u = np.array(y_train_u)

        weix=client.training(X_train_u,y_train_u,global_weights)
        weix=client.scale_model_weights(weix,factors,u.num)
        weit.append(weix)
        
      global_weight=server.sum_scaled_weights(weit) # fedavg
      global_model.set_weights(global_weight)
      loss,acc=Model().evaluate_model(global_model,X_test_server,y_test_server)
      losses.append(loss)
      accuracy.append(acc)

    server_losses.append(losses)
    server_accuracy.append(accuracy)

  return server_losses, server_accuracy

