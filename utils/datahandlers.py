import os
import shutil
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image


class ImageDatasetFolder(Dataset):
        
    def __init__(self, data_dir, data_obj):
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_obj = data_obj
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        self.data_dir = data_dir
        self.image_dataset = datasets.ImageFolder(
            root = os.path.join(self.data_dir, self.data_obj),
            transform = self.data_transforms[self.data_obj])
    
       
    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        return image, label
    
    def __len__(self):
        return len(self.image_dataset)
    
    def num_classes(self):
        return len(self.image_dataset.classes)


class ImageDataset(Dataset):
        
    def __init__(self, image_paths, image_labels, class_to_idx, data_obj):
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_obj = data_obj
        self.class_to_idx = class_to_idx
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        labels_factorized = image_labels.replace(self.class_to_idx)

        self.image_paths = image_paths.values
        self.label_values = torch.as_tensor(labels_factorized.values)
        self.classes = list(self.class_to_idx.keys())
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.data_transforms[self.data_obj](image)
        label = self.label_values[idx].item()
        return image, label
    
    def __len__(self):
        return len(self.image_paths)
    
    def num_classes(self):
        return len(self.classes)


def format_file_path(file_path, directory_name):
    # Get the image filename
    file_name = file_path.split(directory_name)[-1]
    
    # Merge the image filename with the directory
    sub_path = directory_name + file_name
    
    # Replace "%5C" characters with "/"
    sub_path = sub_path.replace('%5C', '/')
    
    return sub_path


def prepare_datafile(csv_file, local_image_folder):
    """
        This Function will prepare the image csv file for the training, validation,
        and testing loops for the Image Classifier.

        It accepts the csv file exported from Label Studio, cleans the file path
        and returns a dataframe with the file path and file label.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, usecols=['image', 'choice'])

    # Formats the images filepaths
    df['image'] = [format_file_path(i,local_image_folder) for i in df['image']]
    
    return df

def get_images_from_source(source_folder, destination_folder, num_images=500):
    # Get a list of all image files in the source folder
    image_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Move the first num_images images to the destination folder
    for i in range(num_images):
        image_file = image_files[i]
        shutil.move(image_file, os.path.join(destination_folder, os.path.basename(image_file)))

    print(f"Moved {num_images} images from {source_folder} to {destination_folder}")
