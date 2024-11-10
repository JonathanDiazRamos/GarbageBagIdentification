import torch
import torchvision.transforms as TF
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import os
import random

# Image directories for training and testing data
garbage_direct = "/test/dataset/garbageBag"
no_garbage_direct = "/test/dataset/BGarbage"


# Dataset statistics for normalization
mean = [0.4427, 0.4489, 0.4485]
std = [0.2141, 0.2181, 0.2364]

# Define various image transformations for data augmentation
None_transforms = TF.Compose([
    TF.ToPILImage(),
    TF.ToTensor(),
    TF.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

hflip_transforms = TF.Compose([
    TF.ToPILImage(),
    TF.RandomHorizontalFlip(p=1),
    TF.ToTensor(),
    TF.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

dis_transforms = TF.Compose([
    TF.ToPILImage(),
    TF.RandomPerspective(distortion_scale=0.5, p=1),
    TF.ToTensor(),
    TF.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

blur_transforms = TF.Compose([
    TF.ToPILImage(),
    TF.GaussianBlur(kernel_size=(19, 23), sigma=(20, 25)),
    TF.ToTensor(),
    TF.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

rotate_transforms = TF.Compose([
    TF.ToPILImage(),
    TF.RandomRotation(15, expand=False),
    TF.ToTensor(),
    TF.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Composite transforms with multiple augmentations
HflipBlur_transforms = TF.Compose([
    TF.ToPILImage(),
    TF.RandomHorizontalFlip(p=1),
    TF.GaussianBlur(kernel_size=(19, 23), sigma=(20, 25)),
    TF.ToTensor(),
    TF.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

HflipRotateDis_transforms = TF.Compose([
    TF.ToPILImage(),
    TF.RandomHorizontalFlip(p=1),
    TF.RandomRotation(15, expand=False),
    TF.RandomPerspective(distortion_scale=0.5, p=1),
    TF.ToTensor(),
    TF.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Function to load images from a directory and apply transformations
def data_set(path, my_transforms):
    train = []
    for img in os.listdir(path):  # Loop through all files in the directory
        img_path = os.path.join(path, img)  # Construct full image path
        image = io.imread(img_path)  # Read image using skimage
        img_tensor = my_transforms(image)  # Apply transformations
        train.append(img_tensor)  # Add transformed image to train list
    return train

# Apply a list of transformations to a dataset of images
def apply_transform(data_path, transform_list):
    image_matrix = []
    for path in data_path:
        for transform in transform_list:
            img_matrix = data_set(path, transform)
            for img in img_matrix:
                # Convert tensor to numpy array and add to image matrix
                image_matrix.append(img.cpu().detach().numpy().T[:, :, :3])
    return np.array(image_matrix)

# Function to assign labels to two arrays, then shuffle the combined list
def label(array1, label1, array2, label2):
    list1 = [[array2[i], label2] for i in range(len(array2))]
    for i in range(len(array1)):
        list1.append([array1[i], label1])
    random.shuffle(list1)  # Shuffle data and labels

    train1, train2 = [], []
    for item in list1:
        train1.append(item[0])  # Images
        train2.append(item[1])  # Labels

    return np.array(train1), np.array(train2)

# Define directories for training and testing images
garbage_test_path = [garbage_direct]
street_test_path = [no_garbage_direct]

# Transformation lists for different datasets
garbage_bag_transform = [
    None_transforms, hflip_transforms, dis_transforms, blur_transforms,
    rotate_transforms, HflipBlur_transforms, HflipRotateDis_transforms
]
street_transform = [None_transforms, hflip_transforms, blur_transforms, HflipBlur_transforms]

# Apply transformations to test datasets
garbage_test_set = apply_transform(garbage_test_path, garbage_bag_transform)
street_test_set = apply_transform(street_test_path, street_transform)

# Print statistics for transformed test datasets
print("Total number of test garbage bag images with transforms:", garbage_test_set.shape[0])
print("Total number of test street images with transforms:", street_test_set.shape[0])
print("Difference between test garbage and street images:", abs(garbage_test_set.shape[0] - street_test_set.shape[0]), "\n")
