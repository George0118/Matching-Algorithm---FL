import tensorflow as tf
import keras
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Rescaling
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from keras.models import Sequential
from keras.regularizers import L2
from keras.applications import EfficientNetV2S, MobileNetV3Small, InceptionV3, DenseNet201, NASNetMobile, MobileNetV2
from keras.applications import inception_v3, mobilenet_v2, densenet, nasnet

class Model:

  def __init__(self):
    pass

  def global_model(self, shape):

    regularization_strength = 0.1

    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(256, activation='relu', kernel_regularizer=L2(regularization_strength)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

  def evaluate_model(self,model, features, labels):
    model.compile(optimizer='rmsprop',
      loss=keras.losses.BinaryCrossentropy(),
      metrics=[keras.metrics.BinaryAccuracy(name="acc")]
    )
    
    score=model.evaluate(features, labels)
    print("Test Loss:", score[0])
    print('Test Accuracy:', score[1])
    return score[0],score[1]
  
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
  
  def base_model(self, model):
     # Define the input layer
    input = tf.keras.Input(shape=(224, 224, 3))

    # Data Augmentation
    # input = self.data_augmentation(input)

    if model == "EfficientNetV2S":
      baseModel = EfficientNetV2S(weights="imagenet", include_top=False, input_tensor=input)
    elif model == "MobileNetV2":
      input = mobilenet_v2.preprocess_input(input)
      baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input)
    elif model == "MobileNetV3Small":
      baseModel = MobileNetV3Small(weights="imagenet", include_top=False, input_tensor=input)
    elif model == "InceptionV3":
      input = inception_v3.preprocess_input(input)
      baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=input)
    elif model == "DenseNet201":
      input = densenet.preprocess_input(input)
      baseModel = DenseNet201(weights="imagenet", include_top=False, input_tensor=input)
    elif model == "NASNetMobile":
      input = nasnet.preprocess_input(input)
      baseModel = NASNetMobile(weights="imagenet", include_top=False, input_tensor=input)
    else:
      return None
  
    for layer in baseModel.layers:
      layer.trainable = False
    
    model = keras.models.Model(inputs=input, outputs=baseModel.output)

    return model
     
  
  def extract_features(self, baseModel, dataset):

    extracted_features = baseModel.predict(dataset)

    return extracted_features
     