from catalyst.data import BalanceClassSampler
import torch
import pandas as pd
from torch.utils.data import DataLoader

from datasets.datasets import get_train_augmentations, Dataset


def get_train_dataloader(df: pd.DataFrame, configs: dict):
    mean = (configs['mean']['r'], configs['mean']['g'], configs['mean']['b'])
    std = (configs['std']['r'], configs['std']['g'], configs['std']['b'])

    transforms = get_train_augmentations(configs['image_size'], mean=mean, std=std)

    try:
        face_detector = configs['face_detector']
    except AttributeError:
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
