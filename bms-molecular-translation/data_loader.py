from torch.utils import data
from torchvision import transforms
import os
from PIL import Image


def make_datapath_list(img_id_list):
    num_img = len(img_id_list)
    datapath_list = list()
    path = "./data/train/"

    t = 0
    i = 0
    while i < 16 and t < num_img:
        j = 0
        while j < 16 and t < num_img:
            k = 0
            while k < 16 and t < num_img:
                dir_path = os.path.join(path, f"{format(i, 'x')}/{format(j, 'x')}/{format(k, 'x')}")
                num_list_dir = len(os.listdir(dir_path))
                m = 0
                while m < num_list_dir and t < num_img:
                    datapath_list.append(os.path.join(dir_path, img_id_list[t]) + ".png")
                    m += 1
                    t += 1
                k += 1
            j += 1
        i += 1

    return datapath_list


class ImageTransform:
    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class ImageDataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img)

        return img_transformed




