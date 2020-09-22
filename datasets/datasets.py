from typing import Callable
import os
import numpy as np
from PIL import Image
import torch
import logging
import argparse
import yaml
import pandas as pd

from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_augmentations(image_size: int = 224):
    return A.Compose(
        [
            # A.CoarseDropout(20),
            # A.Rotate(30),
            # A.RandomCrop(image_size, image_size, p=0.5),
            A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size, 0),
            # A.Normalize(),
            ToTensor(),
        ]
    )


def get_test_augmentations(image_size: int = 224):
    return A.Compose(
        [
            A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size, 0),
            # A.Normalize(),
            ToTensor(),
        ]
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: "pd.DataFrame",
        root: str,
        transforms: Callable,
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

    def analyze(self):
        for index in range(0, len(self.df)):
            if self.df.iloc[index].target == 0:
                self.metadata["fake_samples"] += 1
            else:
                self.metadata["live_samples"] += 1

        print("Live samples: ", self.metadata["live_samples"])
        print("Fake samples: ", self.metadata["fake_samples"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int):
        path = os.path.join(self.root, self.df.iloc[item].path)

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
    parser.add_argument("-data", "--data_root", required=True, help="Dataset root path.")
    parser.add_argument("-config", "--config_file_path", required=True, help="Config file path.")
    args = parser.parse_args()

    with open(args.config_file_path, 'r') as stream:
        configs = yaml.safe_load(stream)

    df = pd.read_csv(configs['train_df'])
    dataset = Dataset(df, configs['path_root'], face_detector=None,
                      transforms=get_train_augmentations())
    print("Length of dataset: ", len(dataset))
    dataset.analyze()
