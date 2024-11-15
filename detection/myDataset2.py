import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import torchvision.transforms as transforms
import cv2
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset


def default_loader(path1, path2, size=None):
    labels = []
    images = []
    # 读取目录下的所有jpg文件
    files1 = sorted([os.path.join(path1, f) for f in os.listdir(path1) if f.endswith('.jpg')])
    files2 = sorted([os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.jpg')])

    # 确保两个目录下的文件数量相同
    assert len(files1) == len(files2), "The number of files in the two directories must be the same."

    # 限制读取的文件数量
    if size is not None:
        files1 = files1[:size]
        files2 = files2[:size]

    for file1, file2 in zip(files1, files2):
        images.append(Image.open(file1).resize((1024, 1024)))
        labels.append(0)
        images.append(Image.open(file2).resize((1024, 1024)))
        labels.append(1)

    return labels, images


class MyDataset(Dataset):

    def __init__(self, path1, path2, train=True, size=None):
        super(MyDataset, self).__init__()
        if train:
            self.labels, self.images = default_loader(path1, path2, size)
        else:
            self.labels, self.images = default_loader(path1, path2, size)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        image = np.asarray(image)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).div(255.0)
        label = self.labels[index]
        label = int(label)
        return image, label
