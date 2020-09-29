import argparse
import yaml
import datetime
import os
import glob

import utils
from dataset import dataloaders
from models.scan import SCAN

import torch
import torchvision
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Config file path.")
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        configs = yaml.safe_load(stream)

    root_dir = configs['log_dir']
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    version = utils.get_latest_version(root_dir)

    version_directory = root_dir + "version_" + str(version)
    if not os.path.isdir(version_directory):
        os.makedirs(version_directory)

    start = datetime.datetime.now()
    configs['start'] = start
    configs['version'] = version

    with open(version_directory + '/configs.yml', 'w') as outfile:
        yaml.dump(configs, outfile, default_flow_style=False)

    # ========================= End of DevOps ==========================
    # ========================= Start of ML ==========================
    device = configs['device']

    train_df = pd.read_csv(configs['train_df'])
    val_df = pd.read_csv(configs['val_df'])

    train_loader = dataloaders.get_train_dataloader(train_df, configs)

    model = SCAN()
    model.to(device)

    if configs['print_model']:
        print(model)

    print("hi")
