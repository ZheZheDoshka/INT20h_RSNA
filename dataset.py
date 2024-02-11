import settings

import pandas as pd

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import pydicom as dicom
import json

from PIL import Image


class ImageDataset(Dataset):
    """
    Class that creates dataset for training task
    Arguments:
    classification - bool, True - set for classification only, False - for obj detection
    transform - transform for input image
    target_transform - transform for label
    labels - path to csv file with cleaned dataset
    labels_df - dataframe with cleaned dataset
    img_dir - path to directory with training images
    """
    def __init__(self, img_dir, labels_df=None,
                 labels=None, transform=None, target_transform=None,
                 classification=True):
        if labels:
            self.img_labels = pd.read_csv(labels)
        if not isinstance(labels_df, type(None)):
            self.img_labels = labels_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.classification = classification
        self.mult = settings.INPUT_HEIGHT/1024

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, idx):
        img_path = self.img_dir + f'/{self.img_labels["patient_path"][idx]}'
        image = Image.open(img_path)
        target = self.img_labels['target'][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        if not self.classification:
            label = torch.Tensor(self.img_labels.loc[idx, 'label']).to(torch.int64)
            bbox = torch.Tensor(self.img_labels.loc[idx, 'bbox']).to(torch.float)
            bbox = bbox*self.mult
            target = {}
            target['boxes'] = bbox
            # i hate it here
            target['labels'] = label
            return image, target
        return image, target


class TestDataset(Dataset):
    """
    Class that creates dataset for training task
    Arguments:
    img_df - dataframe with cleaned dataset
    img_dir - path to directory with training images
    transform - transforms for input
    """
    def __init__(self, img_dir, img_df, transform):
        self.transform = transform
        self.img_labels = img_df
        self.img_dir = img_dir
        self.mult = settings.INPUT_HEIGHT/1024

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, idx):
        patient_id = self.img_labels["patientId"][idx]
        img_path = self.img_dir + f'/{self.img_labels["patientId"][idx]}.dcm'
        ds = dicom.dcmread(img_path)
        image = Image.fromarray(ds.pixel_array)
        if self.transform:
            image = self.transform(image)
        return image, patient_id


def create_test_loader(test_df):
    """
    arguments: test_df - dataframe with test data information

    :return: dataset and dataloader for train data
    """
    resize = transforms.Resize(size=224)
    test_transforms = transforms.Compose([resize,
                                          transforms.ToTensor()])
    test_dataset = TestDataset('kaggle_set/stage_2_test_images', img_df=test_df,
                               transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    return test_dataset, test_dataloader


def create_train_loader(png_dir, train_df, train_box_df=None):
    """
    Function that returns dataloaders and datasets for training
    :param png_dir: path to directory with png files of train dataset
    :param train_df: dataframe with information about classification training dataset
    :param train_box_df: dataframe with info about object detection training dataset
    :return: train_dataset, train_dataloader, train_dataset_box, train_dataloader_box
    """
    h_flip = transforms.RandomVerticalFlip()
    v_flip = transforms.RandomHorizontalFlip()
    rotate = transforms.RandomRotation(degrees=25)
    resize = transforms.RandomResizedCrop(size=224, scale=(0.9, 1))
    train_transforms = transforms.Compose([h_flip, v_flip, rotate, resize,
                                           transforms.ToTensor()])
    base_transform = transforms.Compose([
        # adjust_contrast(),
        transforms.ToTensor()
    ])
    train_dataset = ImageDataset(png_dir, labels_df = train_df, transform=train_transforms)#, classification=False)
    train_dataloader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)#, collate_fn=collate_fn)
    if not isinstance(train_box_df, type(None)):
        train_dataset_box = ImageDataset(png_dir, labels_df=train_box_df, transform=base_transform, classification=False)
        train_dataloader_box = DataLoader(train_dataset_box, batch_size=settings.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        return train_dataset, train_dataloader, train_dataset_box, train_dataloader_box
    return train_dataset, train_dataloader


def collate_fn(batch):
    """
    custom collate function to put boxes into tuples
    """
    return tuple(zip(*batch))