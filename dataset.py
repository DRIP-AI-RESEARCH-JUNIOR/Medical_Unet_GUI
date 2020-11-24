import os
import random
from PIL import Image
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class SyntheticCellDataset(Dataset):

    def __init__(
        self,
        images_dir,
        mask_dir,
    ):
        self.image_list = []
        self.mask_list = []
        mask_lst_ref = []

        for img in glob.glob(image_dir + '/*.TIF'):
            self.mask_list.append(img)

        for img in glob.glob(mask_dir + '/*.TIF'):
            self.image_list.append(img)      

    def transform(self, image, mask):
        # Resize
        resize = transforms.Resize(size=(576, 576))
        image = resize(image)
        mask = resize(mask)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        mask =  Image.open(self.mask_list[idx])

        x, y = self.transform(image, mask)

        # return tensors
        return x, y