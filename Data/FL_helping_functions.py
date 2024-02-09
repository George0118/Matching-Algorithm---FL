from Data.Classes.Model import Model
from Data.Classes.Client import Client

def extract_features_wrapper(user_num, base_model, X_train, user_features):
    user_features[user_num] = Model().extract_features(base_model, X_train)

def training_wrapper(lr, epoch, user_num, user_features, y_train, global_weights, class_weights, factors, weit):
    print("User ", user_num, ":")
    client = Client(lr, epoch, user_num)

    weix = client.training(user_features,
                           y_train,
                           global_weights,
                           class_weights,
                           user_features.shape[1:]
                           )
    weix = client.scale_model_weights(weix, factors)
    weit.append(weix)