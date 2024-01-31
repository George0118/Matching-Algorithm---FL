import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Rescaling
from keras.layers import RandomFlip
from keras.layers import RandomRotation
from keras.layers import BatchNormalization
from keras.layers import SeparableConv2D
from keras.layers import GlobalAveragePooling2D
from keras.regularizers import L2
from keras.applications import VGG16

class Model:

  def __init__(self):
    pass


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

  def global_model(self):

    regularization_strength = 0.01

    # Define the input layer
    input = tf.keras.Input(shape=(224, 224, 3))

    # Data Augmentation
    x = self.data_augmentation(input)

    # Rescaling
    x = Rescaling(1./255)(x)

    # =========== Normal Small Model ============ #

    # # Convolutional layers
    # x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2(regularization_strength))(input)
    # x = MaxPooling2D(2, 2)(x)

    # x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2(regularization_strength))(x)
    # x = MaxPooling2D(2, 2)(x)

    # x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2(regularization_strength))(x)
    # x = MaxPooling2D(2, 2)(x)

    # x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2(regularization_strength))(x)
    # x = MaxPooling2D(2, 2)(x)

    # x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=L2(regularization_strength))(x)
    # x = MaxPooling2D(2, 2)(x)

    # # Flatten the results to feed into a DNN
    # x = Flatten()(x)

    # # Dense layer
    # x = Dense(512, activation='relu', kernel_regularizer=L2(regularization_strength))(x)
    # x = Dropout(0.5)(x)

    # # Output layer
    # output = Dense(1, activation='sigmoid')(x)

    # # Create the model
    # model = keras.Model(inputs=input, outputs=output)

    # ================================================= #

    # ================ VGG16 Version ================= #

    baseModel = VGG16(weights="imagenet", include_top=False,
	                    input_tensor=x)
    
    for layer in baseModel.layers:
            layer.trainable = False
    
    x = baseModel.output

    x = Flatten()(x)
    x = Dense(512, kernel_regularizer=L2(regularization_strength))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1)(x)
    output = Activation('sigmoid')(x)

    model = keras.models.Model(inputs=input, outputs=output)

    # ================================================= #

    return model

  def evaluate_model(self,model,test_X, test_y):
    model.compile(optimizer='rmsprop',
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")]
    )
    score=model.evaluate(test_X, test_y)
    print("Test Loss:", score[0])
    print('Test Accuracy:', score[1])
    return score[0],score[1]