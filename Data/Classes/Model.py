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

class Model:
  
  def __init__(self):
    pass

  # def global_models(self):
  #   model= Sequential([
  #         Flatten(input_shape=(224, 224, 3)),
  #         Dense(128, activation='relu'),
  #         Dropout(0.2),
  #         Dense(10, activation='softmax')
  #     ])
  #   return model
  
  def global_model(self):

    model = Sequential()

    model.add(Rescaling(1./255, input_shape=(224, 224, 3)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

  def evaluate_model(self,model,test_X, test_y):
    model.compile(optimizer='sgd',
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    score=model.evaluate(test_X, test_y)
    print("Test Loss:", score[0])
    print('Test Accuracy:', score[1])
    return score[0],score[1]