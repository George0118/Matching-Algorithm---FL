# Input paths for all disasters

earthquake_input_paths = [
    "../data/cyclone-wildfire-flood-earthquake-database/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database/Earthquake",
    "../data/disasters-dataset-final/disasters/earthquake",
    "../data/disasters-intensity/Disasters/Disasters/Train/earthquake",
    "../data/disasters-intensity/Disasters/Disasters/Test/earthquake",
    "../data/turkiye-earthquake-2023-damaged-buildings/damaged"
]

fire_input_paths = [
    "../data/cyclone-wildfire-flood-earthquake-database/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database/Wildfire",
    "../data/disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Fire_Disaster/Urban_Fire",
    "../data/disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Fire_Disaster/Wild_Fire",
    "../data/disasters-dataset-final/disasters/wildfire",
    "../data/disasters-intensity/Disasters/Disasters/Test/fire",
    "../data/disasters-intensity/Disasters/Disasters/Train/fire",
    "../data/forest-fire-image-dataset/FOREST_FIRE_DATASET/FIRE",
    "../data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/test/fire/Both_smoke_and_fire",
    "../data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/test/fire/Smoke_from_fires",
    "../data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/train/fire/Both_smoke_and_fire",
    "../data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/train/fire/Smoke_from_fires",
    "../data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/val/fire/Both_smoke_and_fire",
    "../data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/val/fire/Smoke_from_fires"
]

flood_input_paths = [
    "../data/cyclone-wildfire-flood-earthquake-database/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database/Flood",
    "../data/disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Water_Disaster",
    "../data/disasters-dataset-final/disasters/flood",
    "../data/disasters-intensity/Disasters/Disasters/Test/flood",
    "../data/disasters-intensity/Disasters/Disasters/Train/flood",
    "../data/flooding-image-dataset/Flood Images"  
]

import cv2
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(file_paths, disaster, test_size=0.2, random_state=42):
    images_for_each_disaster = 10000 * random.uniform(1.1,1.3)
    images = []
    labels = []

    count = 0

    for path in file_paths:
        path = path.replace("../data", "/kaggle/input/custom-disaster-dataset")     # Uncomment when running on kaggle
        for root, dirs, files in os.walk(path):
            for file in files:
                if(count >= images_for_each_disaster):
                    return train_test_split(np.array(images), np.array(labels), test_size=test_size, random_state=random_state)

                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                images.append(image)
                labels.append(disaster)
                count += 1

    return train_test_split(np.array(images), np.array(labels), test_size=test_size, random_state=random_state)