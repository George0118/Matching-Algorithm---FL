from general_parameters import *
import numpy as np
from Classes.Server import Server
from Data.load_images import fire_input_paths, earthquake_input_paths, flood_input_paths, count_images
from general_parameters import *

def dataset_sizes(s: Server, users, cps, total_images):
    # For each server count the images and select appropriate number of images to distribute
    if(s.num == 0): 
        image_num = count_images(fire_input_paths)
        ratio = image_num/total_images
        ratio = 1-np.sqrt(ratio)
        image_num = int(1.7*ratio*image_num)
        img_per_usr = image_num/N_max
    elif(s.num == 1):
        image_num = count_images(flood_input_paths)
        ratio = image_num/total_images
        ratio = 1-np.sqrt(ratio)
        image_num = int(1.3*ratio*image_num)
        img_per_usr = image_num/N_max
    else:
        image_num = count_images(earthquake_input_paths)
        ratio = image_num/total_images
        ratio = 1-np.sqrt(ratio)
        image_num = int(0.8*ratio*image_num)
        img_per_usr = image_num/N_max

    # For each user calculate the minimum distance from the relevant Critical Points
    user_min_distances = [-1]*N
    ratios = [0]*N
    
    for u in users:
        for cp in cps:
            user_x, user_y, user_z = u.x, u.y, u.z
            cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

            distance = np.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

            if(distance < user_min_distances[u.num] or user_min_distances[u.num] == -1):
                user_min_distances[u.num] = distance

    # Calculate the data size ratios based on the user minimum distance from the CPs
    for i in range(N):
        if user_min_distances[i] <= 0.4:
            ratios[i] = 1/(user_min_distances[i] + 1e-6)
        else:
            ratios[i] = 0

    ratios = [ratio/max(ratios) for ratio in ratios]

    # Get Sizes
    sizes = [int(1.8 * np.sqrt(np.sqrt(ratio)) * img_per_usr) for ratio in ratios]

    if sum(sizes) > image_num:
        temp_total = sum(sizes)
        sizes = [size*image_num/temp_total for size in sizes]

    print("Sizes:")
    print(sizes)

    return sizes



def channel_gain(distance, num_user, num_server):   # Channel gain calculation
    g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[num_server][num_user]

    g = 10**(-g / 10)

    return g