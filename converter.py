import pandas as pd
import cv2
import pydicom as dicom

import os
import copy

import settings

INPUT_HEIGHT = settings.INPUT_HEIGHT

def return_train_df(set_path):
    train_df = pd.read_csv(set_path)
    train_df["patient_path"] = train_df.apply(image_path, axis=1)
    train_label_set = copy.deepcopy(train_df[['patient_path', 'x', 'y', 'width', 'height']])
    train_label_set['label'] = copy.deepcopy(train_df['Target'])
    train_label_set_clean = copy.deepcopy(train_label_set)
    train_label_set_clean["x2"] = train_label_set_clean.apply(image_x2, axis=1)
    train_label_set_clean["y2"] = train_label_set_clean.apply(image_y2, axis=1)
    train_label_set_clean['bbox'] = train_label_set_clean[['x', 'y', 'x2', 'y2']].apply(list, axis=1).fillna(0)
    train_label_set_clean = train_label_set_clean.drop(columns=['x', 'y', 'x2', 'y2', 'height', 'width']).groupby(
        'patient_path', as_index=False).agg(list)

    train_label_set_clean["target"] = train_label_set_clean.apply(image_target_label, axis=1)
    return train_label_set_clean




def return_test_df(path, df=None):
    if not isinstance(df, type(None)):
        return df

    df = [f[:-4] for f in os.listdir(path)]
    df = pd.DataFrame({'patientId': df})
    return df


def convert_dcm_to_png(dcm_path, png_path):
    """
    Converts dcm format files to png
    Arguments:
    dcm_path - path to folder with .dcm images
    png_path - path to folder with .png images
    """
    images_path = os.listdir(dcm_path)
    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(dcm_path, image))
        np_im = ds.pixel_array
        image = image.replace('.dcm', '.png')
        np_im = cv2.resize(np_im, (INPUT_HEIGHT, INPUT_HEIGHT))
        cv2.imwrite(os.path.join(png_path, image), np_im)


def image_path(row):
    # returns {patientId}.png
    return f'{row["patientId"]}.png'


def image_target_label(row):
    # returns label from list of labels
    return int(row['label'][0])


def image_x2(row):
    # returns x2 position
    return row['x']+row['width']


def image_y2(row):
    # returns y2 position
    return row['y']+row['height']