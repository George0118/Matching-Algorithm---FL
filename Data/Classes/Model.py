import tensorflow as tf
import keras
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.models import Sequential
from keras.regularizers import L2
from keras.applications import MobileNetV3Large, MobileNetV3Small

class Model:

  def __init__(self):
    pass


  def global_model(self, input_shape):

    input = Input(shape=input_shape)
    x = GlobalMaxPooling2D()(input)
    x = Dense(64, activation='relu')(x)
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

    score = model.evaluate(features, labels)
    print("Test Loss:", score[0])
    print('Test Accuracy:', score[5])
    return score[0],score[5]
  
  
  # ======== Data Augmentation ========= #

  def data_augmentation(self, images):
      data_augmentation_layers = [
        RandomFlip("horizontal"),
        RandomRotation(0.1),
      ]

      for layer in data_augmentation_layers:
          images = layer(images)
      return images

  # ===================================== #

  
  def base_model(self):
     # Define the input layer
    input = tf.keras.Input(shape=(224, 224, 3))

    # Data Augmentation
    # input = self.data_augmentation(input)

    baseModel = MobileNetV3Small(weights="imagenet", include_top=False, input_tensor=input)

    for layer in baseModel.layers:
      layer.trainable = False
    
    model = keras.models.Model(inputs=input, outputs=baseModel.output)

    return model
     
  
  def extract_features(self, baseModel, dataset):

    extracted_features = baseModel.predict(dataset)

    return extracted_features
     