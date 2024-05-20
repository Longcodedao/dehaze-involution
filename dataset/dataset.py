import os
import torch 
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class FoggyCityscape(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.haze_dir = os.path.join(data_dir, "hazy")
        self.clean_dir = os.path.join(data_dir, "clean")

        # Sort for consistency
        self.haze_images = sorted(os.listdir(self.haze_dir))  
        self.clean_images = sorted(os.listdir(self.clean_dir))  

        # Error handling if number of haze and clear images don't match
        if len(self.haze_images) != len(self.clean_images):
            raise ValueError("Number of haze images does not match number of clear images.")

    def __len__(self):
        return len(self.haze_images)

    def __getitem__(self, idx):
        haze_image_name = self.haze_images[idx]
        clean_image_name = self.clean_images[idx]
        haze_image_path = os.path.join(self.haze_dir, haze_image_name)
        clean_image_path = os.path.join(self.clean_dir, clean_image_name)

        haze_image = Image.open(haze_image_path)
        clean_image = Image.open(clean_image_path)

        if self.transform:
            haze_image = self.transform(haze_image)
            clean_image = self.transform(clean_image)
            
        return haze_image, clean_image


class OHaze(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.haze_dir = os.path.join(data_dir, "hazy")
        self.clean_dir = os.path.join(data_dir, "GT")

        # Sort for consistency
        self.haze_images = sorted(os.listdir(self.haze_dir))  
        self.clean_images = sorted(os.listdir(self.clean_dir))  

        # Error handling if number of haze and clear images don't match
        if len(self.haze_images) != len(self.clean_images):
            raise ValueError("Number of haze images does not match number of clear images.")

    def __len__(self):
        return len(self.haze_images)

    def __getitem__(self, idx):
        haze_image_name = self.haze_images[idx]
        clean_image_name = self.clean_images[idx]
        haze_image_path = os.path.join(self.haze_dir, haze_image_name)
        clean_image_path = os.path.join(self.clean_dir, clean_image_name)

 
        haze_image = Image.open(haze_image_path)
        clean_image = Image.open(clean_image_path)

        if self.transform:
            haze_image = self.transform(haze_image)
            clean_image = self.transform(clean_image)

        return haze_image, clean_image
    

class DenseHaze(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.haze_dir = os.path.join(data_dir, "hazy")
        self.clean_dir = os.path.join(data_dir, "GT")

        # Sort for consistency
        self.haze_images = sorted(os.listdir(self.haze_dir))  
        self.clean_images = sorted(os.listdir(self.clean_dir))  

        # Error handling if number of haze and clear images don't match
        if len(self.haze_images) != len(self.clean_images):
            raise ValueError("Number of haze images does not match number of clear images.")

    def __len__(self):
        return len(self.haze_images)

    def __getitem__(self, idx):
        haze_image_name = self.haze_images[idx]
        clean_image_name = self.clean_images[idx]
        haze_image_path = os.path.join(self.haze_dir, haze_image_name)
        clean_image_path = os.path.join(self.clean_dir, clean_image_name)

 
        haze_image = Image.open(haze_image_path)
        clean_image = Image.open(clean_image_path)

        if self.transform:
            haze_image = self.transform(haze_image)
            clean_image = self.transform(clean_image)

        return haze_image, clean_image