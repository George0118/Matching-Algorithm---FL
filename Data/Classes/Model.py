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


class Model:
  
  def __init__(self):
    pass
  
  def global_model(self):

    # Input layer
    input = keras.Input(shape=(224,224,3))

    # Data augmentation
    x = RandomFlip("horizontal")(input)
    x = RandomRotation(0.1)(x)

    x = Rescaling(1./255)(x)

    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model

  def evaluate_model(self,model,test_X, test_y):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    score=model.evaluate(test_X, test_y)
    print("Test Loss:", score[0])
    print('Test Accuracy:', score[1])
    return score[0],score[1]