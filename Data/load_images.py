# Input paths for all disasters

earthquake_input_paths = [
    "../data/cyclone-wildfire-flood-earthquake-database/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone_Wildfire_Flood_Earthquake_Database/Earthquake",
    "../data/disasters-dataset-final/disasters/earthquake",
    "../data/disasters-intensity/Disasters/Disasters/Train/earthquake",
    "../data/disasters-intensity/Disasters/Disasters/Test/earthquake",
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
    "../data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/train/fire/Both_smoke_and_fire",
    "../data/the-wildfire-dataset/the_wildfire_dataset/the_wildfire_dataset/val/fire/Both_smoke_and_fire",
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
from config import N,S
from general_parameters import N_neutral
import random
ran = random.Random(42)

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

    # Shuffle images and labels in parallel
    combined = list(zip(images, labels))
    ran.shuffle(combined)
    images[:], labels[:] = zip(*combined)

    selected_images = [images[i:i + N_neutral] for i in range(0, len(images), N_neutral)]
    selected_labels = [labels[i:i + N_neutral] for i in range(0, len(labels), N_neutral)]

    # Selecting next 250 images and labels for server set
    remaining_images = images[N * N_neutral:]
    remaining_labels = labels[N * N_neutral:]
    server_images = [remaining_images[i:i + N_neutral] for i in range(0, len(remaining_images), N_neutral)]
    server_labels = [remaining_labels[i:i + N_neutral] for i in range(0, len(remaining_labels), N_neutral)]

    return selected_images[:N], server_images[0], selected_labels[:N], server_labels[0], image_num

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

    # Shuffle images and labels in parallel
    combined = list(zip(images, labels))
    ran.shuffle(combined)
    images[:], labels[:] = zip(*combined)

    # Split the images and labels into N lists, each containing N_neutral images and labels
    neutral_image_lists = [images[i:i+N_neutral] for i in range(0, len(images), N_neutral)]
    neutral_label_lists = [labels[i:i+N_neutral] for i in range(0, len(labels), N_neutral)]

    # Create S lists, each containing 250 images and labels, from the remaining images
    remaining_images = images[N * N_neutral:]
    remaining_labels = labels[N * N_neutral:]
    server_image_lists = [remaining_images[i:i+N_neutral] for i in range(0, len(remaining_images), N_neutral)]
    server_label_lists = [remaining_labels[i:i+N_neutral] for i in range(0, len(remaining_labels), N_neutral)]

    return neutral_image_lists[:N], neutral_label_lists[:N], server_image_lists[:S], server_label_lists[:S]