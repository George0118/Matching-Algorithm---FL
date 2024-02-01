from Data.load_images import load_images
from Data.load_images import earthquake_input_paths, fire_input_paths, flood_input_paths
from sklearn.utils import shuffle
import numpy as np
import math

# to get the training data, and split the data via the number of clients
class Get_data:
  def __init__(self, users, servers):
    self.users = users    # clients
    self.servers = servers    # servers
    self.n = len(users)   # num of clients

  def load_data(self):

    # Load and shuffle fire data
    print("Loading Images for Fires...")
    X_train_fire, X_test_fire, y_train_fire, y_test_fire = load_images(fire_input_paths, "fire")
    X_train_fire, y_train_fire = shuffle(X_train_fire, y_train_fire, random_state=42)
    X_test_fire, y_test_fire = shuffle(X_test_fire, y_test_fire, random_state=42)
    print("Fire Images loaded!\n")

    # Load and shuffle flood data
    print("Loading Images for Floods...")
    X_train_flood, X_test_flood, y_train_flood, y_test_flood = load_images(flood_input_paths, "flood")
    X_train_flood, y_train_flood = shuffle(X_train_flood, y_train_flood, random_state=42)
    X_test_flood, y_test_flood = shuffle(X_test_flood, y_test_flood, random_state=42)
    print("Flood Images loaded!\n")

    # Load and shuffle earthquake data
    print("Loading Images for Earthquakes...")
    X_train_earthquake, X_test_earthquake, y_train_earthquake, y_test_earthquake = load_images(earthquake_input_paths, "earthquake")
    X_train_earthquake, y_train_earthquake = shuffle(X_train_earthquake, y_train_earthquake, random_state=42)
    X_test_earthquake, y_test_earthquake = shuffle(X_test_earthquake, y_test_earthquake, random_state=42)
    print("Earthquake Images loaded!\n")

    return (X_train_fire, X_test_fire, y_train_fire, y_test_fire) ,\
           (X_train_flood, X_test_flood, y_train_flood, y_test_flood),\
           (X_train_earthquake, X_test_earthquake, y_train_earthquake, y_test_earthquake)


  def split_data(self, data, server): 
    users = self.users
    critical_points = server.get_critical_points()
    user_avg_distances = [0]*len(users)

    for u in users:
      for cp in critical_points:
        user_x, user_y, user_z = u.x, u.y, u.z
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

        user_avg_distances[u.num] += distance

      user_avg_distances[u.num] /= len(critical_points) 

    sizes = [0]*len(users)

    for i in range(len(users)):
      sizes[i] = int(len(data)*math.sqrt(1/user_avg_distances[i])/len(users))

    total_size = sum(sizes)
    sizes = [size * len(data) // total_size for size in sizes]

    s_data = []
    index = 0
    
    for size in sizes:
        c_data = data[index:index + size]
        s_data.append(c_data)
        index += size
    
    return s_data

  def pre_data(self):

    (X_train_fire, X_test_fire, y_train_fire, y_test_fire) ,\
    (X_train_flood, X_test_flood, y_train_flood, y_test_flood),\
    (X_train_earthquake, X_test_earthquake, y_train_earthquake, y_test_earthquake) = self.load_data()

    # Split fire data to each user
    X_train_fire=self.split_data(X_train_fire, self.servers[0]) 
    y_train_fire=self.split_data(y_train_fire, self.servers[0])
    X_test_fire=self.split_data(X_test_fire, self.servers[0]) 
    y_test_fire=self.split_data(y_test_fire, self.servers[0])

    # Split flood data to each user
    X_train_flood=self.split_data(X_train_flood, self.servers[1]) 
    y_train_flood=self.split_data(y_train_flood, self.servers[1])
    X_test_flood=self.split_data(X_test_flood, self.servers[1]) 
    y_test_flood=self.split_data(y_test_flood, self.servers[1])

    # Split earthquake data to each user
    X_train_earthquake=self.split_data(X_train_earthquake, self.servers[2]) 
    y_train_earthquake=self.split_data(y_train_earthquake, self.servers[2])
    X_test_earthquake=self.split_data(X_test_earthquake, self.servers[2]) 
    y_test_earthquake=self.split_data(y_test_earthquake, self.servers[2])

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
  
  