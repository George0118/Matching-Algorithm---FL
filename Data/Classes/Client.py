from Data.Classes.Model import Model
from keras.optimizers import Adam
import keras

class Client:
  
  def __init__(self,lr,epoch,u_num):
    self.u_num = u_num
    self.epoch=epoch
    self.lr = lr
    self.loss=keras.losses.BinaryCrossentropy(),
    self.metrics=['accuracy']
    self.optimizer = Adam(learning_rate=self.lr)
    
  def weight_client(self,data,m,n):
    wei_client = []
    for i in range(n):
        len_data = len(data[i])
        proba = len_data / m
        wei_client.append(proba)
    return wei_client
    
  
  def scale_model_weights(self,weights,scalar):
    '''function for scaling a models weights'''
    scaled_weights = [scalar * w for w in weights]
      
    return scaled_weights

  def training(self, features, labels, global_weights, shape):
    model = Model().global_model(shape)

    model.compile(optimizer=self.optimizer,
                  loss=self.loss,
                  metrics=[
                      keras.metrics.TruePositives(name='tp'),
                      keras.metrics.FalsePositives(name='fp'),
                      keras.metrics.TrueNegatives(name='tn'),
                      keras.metrics.FalseNegatives(name='fn'),
                      keras.metrics.BinaryAccuracy(name='accuracy'),
                  ]
                  )
    model.set_weights(global_weights)

    history = model.fit(features, labels, epochs=self.epoch, verbose=2)
    print()

    weights = model.get_weights()
    loss = history.history['loss'][-1]  # Get the final training loss
    accuracy = history.history['accuracy'][-1]  # Get the final training accuracy

    return weights, loss, accuracy
  
  def evaluate(self, features, labels, weights, shape):
    model = Model().global_model(shape)

    model.compile(optimizer=self.optimizer,
                  loss=self.loss,
                  metrics=[
                      keras.metrics.TruePositives(name='tp'),
                      keras.metrics.FalsePositives(name='fp'),
                      keras.metrics.TrueNegatives(name='tn'),
                      keras.metrics.FalseNegatives(name='fn'),
                      keras.metrics.BinaryAccuracy(name='accuracy'),
                  ]
                  )
    model.set_weights(weights)

    score = model.evaluate(features, labels, verbose=2)
    print()

    return score[0],score[5]