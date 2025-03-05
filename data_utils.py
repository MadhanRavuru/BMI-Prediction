import csv
import os

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data

  
class BMIDataset(data.Dataset):
    def __init__(self, csv_file, img_folder, scaler, transform=None, use_gender=True):
        
        self.img_folder = img_folder
        self.transform = transform
        self.use_gender = use_gender
        self.scaler = scaler

        self.image_names = []
        self.bmi_values = []
        self.gender_values = []

        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.image_names.append(row['name'])
                self.bmi_values.append(float(row['bmi']))
                self.gender_values.append(row['gender'])
        
        # transform the bmi values
        self.bmi_values = self.scaler.transform(np.array(self.bmi_values).reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.") 
    
    def get_item_from_index(self, idx):
        
        img_name = os.path.join(self.img_folder, self.image_names[idx])
        image = Image.open(img_name).convert('RGB')
        bmi = self.bmi_values[idx]
        gender = self.gender_values[idx] if self.use_gender else None 
        
        if self.transform is not None:
            image = self.transform(image)
        
        if gender in ["Male", "Female", None]: 
            if gender == "Male":
                gender_tensor = torch.tensor([1, 0], dtype=torch.float32)
            elif gender == "Female":
                gender_tensor = torch.tensor([0, 1], dtype=torch.float32)
            else:  # When gender is None
                gender_tensor = torch.tensor([0, 0], dtype=torch.float32)  # Return a default tensor if gender is None 
        else:
            raise ValueError("Invalid gender value. Must be 'Male', 'Female', or None.")
        
        return image, gender_tensor, torch.tensor(bmi, dtype=torch.float32)
