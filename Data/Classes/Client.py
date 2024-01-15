import tensorflow as tf
from Data.Classes.Model import Model
from keras.optimizers.legacy import SGD
import keras

class Client:
  
  def __init__(self,lr,epoch):
    self.epoch=epoch
    self.lr = lr
    self.loss='binary_crossentropy'
    self.metrics = ['accuracy']
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

  def training(self,X,y,global_weights):

    model=Model().global_model()

    model.compile(optimizer=self.optimizer,
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")]
    )
    model.set_weights(global_weights)
    model.fit(X,y,epochs=self.epoch)
    weights=model.get_weights()

    return weights