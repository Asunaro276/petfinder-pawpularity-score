import cv2
import pandas as pd
from tqdm.notebook import tqdm
import glob
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import models, transforms

from util import dicom2array


def make_train_df():
    train_image = pd.read_csv("data/train_image_level.csv")
    train_study = pd.read_csv("data/train_study_level.csv")

    train_dir = "data/train/"
    train_study["StudyInstanceUID"] = train_study["id"].apply(lambda x: x.replace("_study", ""))
    train = train_image.merge(train_study, on="StudyInstanceUID")

    paths = []
    for instance_id in tqdm(train["StudyInstanceUID"]):
        paths.append(glob.glob(os.path.join(train_dir, instance_id + "/*/*"))[0])
    train["path"] = paths
    train = train.drop(["id_x", "id_y"], axis=1)
    return train


class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "val": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)


class SIIMData(Dataset):
    def __init__(self, df, transform=None, phase="train", augments=None, img_size=(400, 400)):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.transform = transform
        self.phase = phase
        self.augments = augments
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id = self.df["StudyInstanceUID"].values[index]

        image_path = self.df["path"].values[index]
        image = dicom2array(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image, self.phase)

        label = self.df[self.df["StudyInstanceUID"] == image_id].values.tolist()[0][3:7]
        return image, torch.argmax(torch.tensor(label))

if __name__ == "__main__":
    train = make_train_df()
    train_dataset = SIIMData(df=train, transform=ImageTransform(224, (450, 450, 450), (200, 200, 200)), phase="train")
    train_dataset.getitem(0)
