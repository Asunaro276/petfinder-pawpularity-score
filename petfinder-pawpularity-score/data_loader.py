import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils import data
from PIL import Image


class ImageTransform:
    def __init__(self, mean, std, size=(256, 256)):
        self.data_transform = transforms.Compose([
            transforms.Normalize(mean, std),
            transforms.Resize(size=size)],
        )

    def __call__(self, img):
        return self.data_transform(img)


class ImageDataset(data.Dataset):
    def __init__(self, data_list, transform, label_list=None):
        self.data_list = data_list
        self.transform = transform
        self.label = label_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = Image.open(img_path)
        img = torch.Tensor(np.array(img))
        img = img.transpose(1, 2).transpose(0, 1)

        img_transformed = self.transform(img)

        if self.label is not None:
            return img_transformed, self.label[index]

        return img_transformed

