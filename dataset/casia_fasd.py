from typing import Callable
import os
import tqdm
import argparse
import yaml
import numpy as np
from PIL import Image
import torch
import torchvision
import logging
import pandas as pd
import time
import matplotlib.pyplot as plt
from catalyst.data import BalanceClassSampler
from torch.utils.data import DataLoader

from dataset.utils import imshow

from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_augmentations(image_size: int = 224, mean: tuple = (0, 0, 0), std: tuple = (1, 1, 1)):
    return A.Compose(
        [
            # A.RandomBrightnessContrast(brightness_limit=32, contrast_limit=(0.5, 1.5)),
            # A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=(1, 2)),
            # A.CoarseDropout(20),
            A.Rotate(10),

            A.Resize(image_size, image_size),
            # A.RandomCrop(image_size, image_size, p=0.5),

            A.LongestMaxSize(image_size),
            A.Normalize(mean=mean, std=std),
            A.HorizontalFlip(),
            A.PadIfNeeded(image_size, image_size, 0),
            # A.Transpose(),
            ToTensor(),
        ]
    )


def get_test_augmentations(image_size: int = 224, mean: tuple = (0, 0, 0), std: tuple = (1, 1, 1)):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.LongestMaxSize(image_size),
            A.Normalize(mean=mean, std=std),
            A.PadIfNeeded(image_size, image_size, 0),
            ToTensor(),
        ]
    )


def get_train_dataloader(df: pd.DataFrame, configs: dict):
    mean = (configs['mean']['r'], configs['mean']['g'], configs['mean']['b'])
    std = (configs['std']['r'], configs['std']['g'], configs['std']['b'])

    transforms = get_train_augmentations(configs['image_size'], mean=mean, std=std)

    try:
        face_detector = configs['face_detector']
    except KeyError:
        face_detector = None
    dataset = Dataset(
        df, configs['path_root'], transforms, face_detector=face_detector
    )
    if configs['use_balance_sampler']:
        labels = list(df.target.values)
        sampler = BalanceClassSampler(labels, mode="upsampling")
        shuffle = False
    else:
        sampler = None
        shuffle = True

    dataloader = DataLoader(
        dataset,
        batch_size=configs['batch_size'],
        num_workers=configs['num_workers_train'],
        sampler=sampler,
        shuffle=shuffle,
    )
    return dataloader


def get_validation_dataloader(df: pd.DataFrame, configs: dict):
    mean = (configs['mean']['r'], configs['mean']['g'], configs['mean']['b'])
    std = (configs['std']['r'], configs['std']['g'], configs['std']['b'])

    transforms = get_train_augmentations(configs['image_size'], mean=mean, std=std)

    try:
        face_detector = configs['face_detector']
    except KeyError:
        face_detector = None
    dataset = Dataset(
        df, configs['path_root'], transforms, face_detector=face_detector
    )

    dataloader = DataLoader(
        dataset,
        batch_size=configs['batch_size'],
        num_workers=configs['num_workers_val'],
        shuffle=False,
    )
    return dataloader


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: "pd.DataFrame",
        root: str,
        transforms: Callable,
        custom_transforms: list = None,
        face_detector: dict = None,
        with_labels: bool = True,
    ):
        self.df = df.sample(frac=1).reset_index(drop=True)
        self.root = root
        self.transforms = transforms
        self.with_labels = with_labels
        self.face_extractor = None
        if face_detector is not None:
            face_detector["keep_all"] = True
            face_detector["post_process"] = False
            self.face_extractor = MTCNN(**face_detector)
        self.metadata = {
            "live_samples": 0,
            "fake_samples": 0
        }

    def analyze(self, n: int = 0):
        if n > 0:
            indices = np.random.randint(len(self.df), size=n)

            result = torch.Tensor()
            for index in indices:
                img, _ = self.__getitem__(index)
                img = img.unsqueeze(0)
                result = torch.cat((result, img), dim=0)

            grid = torchvision.utils.make_grid(result)
            imshow(grid)

        for index in tqdm.tqdm(range(0, len(self.df))):
            if self.df.iloc[index].target == 0:
                self.metadata["fake_samples"] += 1
            else:
                self.metadata["live_samples"] += 1

        print("Live samples: ", self.metadata["live_samples"])
        print("Fake samples: ", self.metadata["fake_samples"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int):
        path = self.df.iloc[item].path

        image = Image.open(path)
        if self.with_labels:
            target = self.df.iloc[item].target

        if self.face_extractor is not None:
            faces, probs = self.face_extractor(image, return_prob=True)
            if faces is None:
                logging.warning(f"{path} doesn't containt any face!")
                image = self.transforms(image=np.array(image))["image"]
                if self.with_labels:
                    return image, target
                else:
                    return image
            if faces.shape[0] != 1:
                logging.warning(
                    f"{path} - {faces.shape[0]} faces detected"
                )
                face = (
                    faces[np.argmax(probs)]
                    .numpy()
                    .astype(np.uint8)
                    .transpose(1, 2, 0)
                )
            else:
                face = faces[0].numpy().astype(np.uint8).transpose(1, 2, 0)
            image = self.transforms(image=face)["image"]
        else:
            image = self.transforms(image=np.array(image))["image"]

        if self.with_labels:
            return image, target
        else:
            return image


if __name__ == "__main__":
    print("Running dataset analysis...")

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config_file_path", required=True, help="Config file path.")
    args = parser.parse_args()

    with open(args.config_file_path, 'r') as stream:
        configs = yaml.safe_load(stream)

    data_df = pd.read_csv(configs['val_df'])
    dataset = Dataset(data_df, configs['path_root'], transforms=get_train_augmentations())
    print("Length of dataset: ", len(dataset))
    dataset.analyze(n=8)
