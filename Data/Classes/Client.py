import tensorflow as tf
from Data.Classes.Model import Model
from keras.optimizers.legacy import SGD
import keras
from sklearn.model_selection import KFold

class Client:
  
  def __init__(self,lr,epoch):
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

  def training(self, X, y, global_weights):
      n_splits = 3

      kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
      cv_weights = []

      print("CrossValidation")

      for train_index, val_index in kf.split(X):

          model = Model().global_model()
          model.compile(optimizer="rmsprop",
                        loss=self.loss,
                        metrics=self.metrics
                        )
          model.set_weights(global_weights)
          model.fit(X[train_index], y[train_index], epochs=self.epoch)

          val_loss = model.evaluate(X[val_index], y[val_index])[0]
          print(f'Validation Loss: {val_loss}')

          cv_weights.append(model.get_weights())

      avg_weights = [sum(weight[i] for weight in cv_weights) / n_splits for i in range(len(cv_weights[0]))]

      return avg_weights