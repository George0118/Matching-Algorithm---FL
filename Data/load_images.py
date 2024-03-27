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

neutral_images_paths = [
    "../data/neutral_images"
]

import cv2
import os
import math
import numpy as np
import random
from config import N,S
from general_parameters import N_neutral

def count_images(input_paths):  # Count images in input paths
    image_num = 0

    for path in input_paths:
        # path = path.replace("../data", "/kaggle/input/custom-disaster-dataset")     # Uncomment when running on kaggle
        files = os.listdir(path)
        image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        image_num += len(image_files)
    
    return image_num

def load_images(file_paths, disaster):

    total_images = count_images(fire_input_paths + flood_input_paths + earthquake_input_paths)
    image_num = count_images(file_paths)
    ratio = image_num/total_images
    ratio = 1-math.sqrt(ratio)
    if disaster == "fire":
        factor = 1.7
    elif disaster == "flood":
        factor = 1.3
    elif disaster == "earthquake":
        factor = 0.8
    image_num = int(factor*ratio*image_num)

    images = []
    labels = []

    for path in file_paths:
        # path = path.replace("../data", "/kaggle/input/custom-disaster-dataset")     # Uncomment when running on kaggle
        for root, dirs, files in os.walk(path):
            for file in files:
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                images.append(image)
                labels.append(disaster)

    selected_indices = random.sample(range(len(images)), image_num)
    selected_images = [images[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]

    # Select another 250 indices for the server
    remaining_indices = set(range(len(images))) - set(selected_indices)
    server_indices = random.sample(remaining_indices, 250)
    server_images = [images[i] for i in server_indices]
    server_labels = [labels[i] for i in server_indices]

    return selected_images, server_images, selected_labels, server_labels

def load_neutral_images():
    images = []
    labels = []

    for path in neutral_images_paths:
        # path = path.replace("../data", "/kaggle/input/custom-disaster-dataset")     # Uncomment when running on kaggle
        for root, dirs, files in os.walk(path):
            for file in files:
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                images.append(image)
                labels.append("neutral")

    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)

    # Split the images and labels into N lists, each containing N_neutral images and labels
    neutral_image_lists = [images[i:i+N_neutral] for i in range(0, len(images), N_neutral)]
    neutral_label_lists = [labels[i:i+N_neutral] for i in range(0, len(labels), N_neutral)]

    # Create S lists, each containing 250 images and labels, from the remaining images
    remaining_images = images[N * N_neutral:]
    remaining_labels = labels[N * N_neutral:]
    server_image_lists = [remaining_images[i:i+250] for i in range(0, len(remaining_images), 250)]
    server_label_lists = [remaining_labels[i:i+250] for i in range(0, len(remaining_labels), 250)]

    return neutral_image_lists[:N], neutral_label_lists[:N], server_image_lists[:S], server_label_lists[:S]