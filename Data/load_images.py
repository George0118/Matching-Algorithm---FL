# Input paths for all disasters

earthquake_input_paths = [
    "./Data/data/cyclone-wildfire-flood-earthquake-database/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database/Earthquake",
    "./Data/data/disasters-dataset-final/disasters/earthquake",
    "./Data/data/disasters-intensity/Disasters/Disasters/Train/earthquake",
    "./Data/data/disasters-intensity/Disasters/Disasters/Test/earthquake",
    "./Data/data/turkiye-earthquake-2023-damaged-buildings/damaged"
]

fire_input_paths = [
    "./Data/data/cyclone-wildfire-flood-earthquake-database/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database/Wildfire",
    "./Data/data/disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Fire_Disaster/Urban_Fire",
    "./Data/data/disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Fire_Disaster/Wild_Fire",
    "./Data/data/disasters-dataset-final/disasters/wildfire",
    "./Data/data/disasters-intensity/Disasters/Disasters/Test/fire",
    "./Data/data/disasters-intensity/Disasters/Disasters/Train/fire",
    "./Data/data/forest-fire-image-dataset/FOREST_FIRE_DATASET/FIRE",
    "./Data/data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/test/fire/Both_smoke_and_fire",
    "./Data/data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/test/fire/Smoke_from_fires",
    "./Data/data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/train/fire/Both_smoke_and_fire",
    "./Data/data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/train/fire/Smoke_from_fires",
    "./Data/data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/val/fire/Both_smoke_and_fire",
    "./Data/data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/val/fire/Smoke_from_fires"
]

flood_input_paths = [
    "./Data/data/cyclone-wildfire-flood-earthquake-database/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database/Flood",
    "./Data/data/disaster-images-dataset/Comprehensive Disaster Dataset(CDD)/Water_Disaster",
    "./Data/data/disasters-dataset-final/disasters/flood",
    "./Data/data/disasters-intensity/Disasters/Disasters/Test/flood",
    "./Data/data/disasters-intensity/Disasters/Disasters/Train/flood",
    "./Data/data/flooding-image-dataset/Flood Images"  
]

images_for_each_disaster = 10000

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(file_paths, disaster, test_size=0.2, random_state=42):
    images = []
    labels = []

    for path in file_paths:
        for root, dirs, files in os.walk(path):
            for file in files:

                if len(images) >= images_for_each_disaster:
                    return train_test_split(np.array(images), np.array(labels), test_size=test_size, random_state=random_state)

                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                images.append(image)
                labels.append(disaster)

    return train_test_split(np.array(images), np.array(labels), test_size=test_size, random_state=random_state)