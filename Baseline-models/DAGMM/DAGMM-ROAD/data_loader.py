import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torchvision import transforms
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
class KDD99Loader(object):
    def __init__(self, train_dataset, mode="train"):
        self.mode=mode
        # data = np.load(data_path)
        #
        # labels = data["kdd"][:,-1]
        # features = data["kdd"][:,:-1]
        # N, D = features.shape
        #
        # normal_data = features[labels==1]
        # normal_labels = labels[labels==1]
        #
        # N_normal = normal_data.shape[0]
        #
        # attack_data = features[labels==0]
        # attack_labels = labels[labels==0]
        #
        # N_attack = attack_data.shape[0]
        #
        # randIdx = np.arange(N_attack)
        # np.random.shuffle(randIdx)
        # N_train = N_attack // 2

        self.train = train_dataset #attack_data[randIdx[:N_train]]
        # self.train_labels = attack_labels[randIdx[:N_train]]
        #
        # self.test = attack_data[randIdx[N_train:]]
        # self.test_labels = attack_labels[randIdx[N_train:]]
        #
        # self.test = np.concatenate((self.test, normal_data),axis=0)
        # self.test_labels = np.concatenate((self.test_labels, normal_labels),axis=0)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return self.train.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index]), np.float32(self.train_labels[index])
        else:
           return np.float32(self.test[index]), np.float32(self.test_labels[index])
        

def get_loader(data_path, batch_size, mode='train'):
    """Build and return data loader."""

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))  # get data root path
    image_path = os.path.join(data_root, "StatGraph/BaselineModels/ROAD9_9_3-each27") #ROAD_small # flower data set path
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, mode),
                                         transform=transform)
    # train_dataset = train_dataset.resize(len(train_dataset[0]), -1)
    # train_dataset = torch.tensor(train_dataset).reshape(len(train_dataset[0]), -1)
    train_num = len(train_dataset)
    print(train_num)

    # # {'normal':0, 'DoS':1, 'Fuzzy':2, 'Gear':3, 'RPM':4}
    nw = 0#min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    KDD99Loader(train_dataset, mode)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),transform=transform)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size, shuffle=False,num_workers=nw)



    # dataset = KDD99Loader(data_path, mode)
    # shuffle = False
    # if mode == 'train':
    #     shuffle = True
    # data_loader = DataLoader(dataset=dataset,  batch_size=batch_size, shuffle=shuffle)
    return train_loader
