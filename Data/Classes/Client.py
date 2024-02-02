import tensorflow as tf
from Data.Classes.Model import Model
from keras.optimizers.legacy import SGD
import keras
from sklearn.model_selection import KFold
from itertools import combinations

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

  def training(self, train_dataset, global_weights, batch_size = 128, n_splits = 3):
      model = Model().global_model()
      model.compile(optimizer="rmsprop",
                    loss=self.loss,
                    metrics=self.metrics
                    )
      model.set_weights(global_weights)

      train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

      split_size = len(train_dataset) // n_splits

      splits = [train_dataset.skip(i * split_size).take(split_size) for i in range(n_splits)]

      splits_combinations = list(combinations(splits, n_splits-1))
      
      cv_weights = []

      print("CrossValidation")

      for sc in splits_combinations:
        train_ds = sc[0]
        for split in sc[1:]:
          train_ds = train_ds.concatenate(split)

        model.fit(train_ds, epochs=self.epoch)

        cv_weights.append(model.get_weights())

      avg_weights = [sum(weight[i] for weight in cv_weights) / n_splits for i in range(len(cv_weights[0]))]

      print()

      return avg_weights