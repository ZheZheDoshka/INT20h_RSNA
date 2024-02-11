import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

import argparse
import collections
collections.Iterable = collections.abc.Iterable


parser = argparse.ArgumentParser(description='Inference')

parser.add_argument('run', choices=['cl', 'frcnn'], help='Create and save model')
parser.add_argument('--path', type=str, default='kaggle_set/stage_2_test_images', help='Path to test images')


def load_model(path_model, model_type, lr, wd=0.0):
    """
    Loads in model
    path_model - path to model
    model_type - builder function
    """
    model = model_type()
    model.load_state_dict(torch.load(path_model))
    return model


def load_models(path_cl, path_det):
    """
    Loads in both models
    path_cl - path to classification model
    path_det - path to detection model
    """
    cl = load_model(path_cl, Classifier, 0.001)
    cl = cl.cuda()
    frcnn = load_model(path_det, FRCNN, 0.0001, wd=0.0005)
    frcnn = frcnn.cuda()
    return cl, frcnn


def FRCNN():
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    return model


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)  # 1 - input channels, 32 - output/filters
        self.dropout = nn.Dropout(p=0.2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        self.dense1 = nn.Linear(25088, 128)  # 112*112*2
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.pool2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = F.relu(self.pool3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.pool4(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        out = x.flatten(start_dim=1)
        return out

def create_save_cl(path):
    # creates and saves classification model
    cl = Classifier()
    torch.save(cl.state_dict(), path)


def create_save_frcnn(path):
    # creates and saves classification model
    frcnn = FRCNN()
    torch.save(frcnn.state_dict(), path)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.run == 'cl':
        create_save_cl(args.path)
    elif args.run == 'frcnn':
        create_save_frcnn(args.path)
    else:
        raise ValueError('Invalid run type, expected "cl" or "frcnn"')
