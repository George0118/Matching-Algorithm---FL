from Data.load_images import load_images
from Data.load_images import earthquake_input_paths, fire_input_paths, flood_input_paths
from sklearn.utils import shuffle
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# to get the training data, and split the data via the number of clients
class Get_data:
  def __init__(self,n):
    self.n=n # number of clients

  def load_data(self):

    # Load and shuffle fire data
    print("Loading Images for Fires\n")
    X_train_fire, X_test_fire, y_train_fire, y_test_fire = load_images(fire_input_paths, "fire")
    X_train_fire, y_train_fire = shuffle(X_train_fire, y_train_fire, random_state=42)
    X_test_fire, y_test_fire = shuffle(X_test_fire, y_test_fire, random_state=42)

    # Load and shuffle flood data
    print("Loading Images for Floods\n")
    X_train_flood, X_test_flood, y_train_flood, y_test_flood = load_images(flood_input_paths, "flood")
    X_train_flood, y_train_flood = shuffle(X_train_flood, y_train_flood, random_state=42)
    X_test_flood, y_test_flood = shuffle(X_test_flood, y_test_flood, random_state=42)

    # Load and shuffle earthquake data
    print("Loading Images for Earthquakes\n")
    X_train_earthquake, X_test_earthquake, y_train_earthquake, y_test_earthquake = load_images(earthquake_input_paths, "earthquake")
    X_train_earthquake, y_train_earthquake = shuffle(X_train_earthquake, y_train_earthquake, random_state=42)
    X_test_earthquake, y_test_earthquake = shuffle(X_test_earthquake, y_test_earthquake, random_state=42)

    return (X_train_fire, X_test_fire, y_train_fire, y_test_fire) ,\
           (X_train_flood, X_test_flood, y_train_flood, y_test_flood),\
           (X_train_earthquake, X_test_earthquake, y_train_earthquake, y_test_earthquake)

  # def flatten_data(self,X):
  #   data=[]
  #   for i in X:
  #     da=i.flatten()
  #     data.append(da)
  #   return data

  # def one_d_to_n_d(self,X):
  #   data=[]
  #   for i in X:
  #     da=i.reshape(224,224)
  #     data.append(da)
  #   return data

  # def non_iid(self,X,y):
  #   train_X=self.flatten_data(X)
  #   train_y=list(y)
  #   train_data=np.c_[train_X,train_y]
  #   sort_data=train_data[np.argsort(train_data[:,224**2])]
  #   train_x=sort_data[:,0:224**2]
  #   train_Y=np.array(sort_data[:,224**2])
  #   train_x_da=np.array(self.one_d_to_n_d(train_x))
  #   return train_x_da,train_Y

  def split_data(self, data): 
    size=int(len(data) / self.n)
    s_data = []
    for i in range(0, int(len(data)) + 1, size):
        c_data = data[i:i + size]
        if c_data.size > 0:
            s_data.append(c_data)
    return s_data

  def pre_data(self):

    (X_train_fire, X_test_fire, y_train_fire, y_test_fire) ,\
    (X_train_flood, X_test_flood, y_train_flood, y_test_flood),\
    (X_train_earthquake, X_test_earthquake, y_train_earthquake, y_test_earthquake) = self.load_data()

    # Split fire data to each user
    X_train_fire=self.split_data(X_train_fire) 
    y_train_fire=self.split_data(y_train_fire)
    X_test_fire=self.split_data(X_test_fire) 
    y_test_fire=self.split_data(y_test_fire)

    # Split flood data to each user
    X_train_flood=self.split_data(X_train_flood) 
    y_train_flood=self.split_data(y_train_flood)
    X_test_flood=self.split_data(X_test_flood) 
    y_test_flood=self.split_data(y_test_flood)

    # Split earthquake data to each user
    X_train_earthquake=self.split_data(X_train_earthquake) 
    y_train_earthquake=self.split_data(y_train_earthquake)
    X_test_earthquake=self.split_data(X_test_earthquake) 
    y_test_earthquake=self.split_data(y_test_earthquake)

    print("Splited Successfully\n")

    # Concatenate training separated data
    X_train = [np.concatenate((X_train_fire[i], X_train_flood[i], X_train_earthquake[i])) for i in range(self.n)]
    y_train = [np.concatenate((y_train_fire[i], y_train_flood[i], y_train_earthquake[i])) for i in range(self.n)]

    # Combine testing data
    X_test = [np.concatenate((X_test_fire[i], X_test_flood[i], X_test_earthquake[i])) for i in range(self.n)]
    y_test = [np.concatenate((y_test_fire[i], y_test_flood[i], y_test_earthquake[i])) for i in range(self.n)]

    # Shuffle the training Data of each user
    for i in range(self.n):
       X_train[i], y_train[i] = shuffle(X_train[i], y_train[i], random_state=42)

    print("Data Concatenated and Shuffled Successfully\n")

    return X_train, y_train, X_test, y_test
  
  