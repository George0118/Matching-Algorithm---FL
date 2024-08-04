from config import num
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(num)  # Use GPU device
import tensorflow as tf
import keras
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.regularizers import L2
from Data.fl_parameters import lr, epoch
from keras.optimizers import Adam

class Model:

  def __init__(self):
    pass


  def global_model(self, input_shape):

    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_output = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=input, outputs=x_output)

    return model
  

  def evaluate_model(self, model, features, labels):
    model.compile(optimizer='adam',
      loss=keras.losses.BinaryCrossentropy(),
      metrics=[
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
      ]
    )

    score = model.evaluate(features, labels, verbose=2)
    print("Test Loss:", score[0])
    print('Test Accuracy:', score[5])
    return score[0],score[5]
  
  def train_model(self, model, features, labels):
    model.compile(optimizer=Adam(lr=lr),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=[
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
      ]
      )

    model.fit(features, labels, epochs=epoch, verbose=2)
    print()

  
  def base_model(self):
     # Define the input layer
    input = tf.keras.Input(shape=(224, 224, 3))

    baseModel = MobileNetV3Large(weights="./Data/weights_mobilenet_v3_large_224_1.0_float_no_top.h5", include_top=False, input_tensor=input)

    for layer in baseModel.layers:
      layer.trainable = False
    
    model = keras.models.Model(inputs=input, outputs=baseModel.output)

    return model
     
  
  def extract_features(self, baseModel, dataset):

    extracted_features = baseModel.predict(dataset)

    return extracted_features
     