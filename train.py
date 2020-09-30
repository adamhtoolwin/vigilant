import argparse
import yaml
import datetime
import os
import glob
import tqdm

import utils
from dataset import dataloaders
from models.scan import SCAN, Vigilant, SCANEncoder
from vigilant import *

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np


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

    weights_directory = version_directory + "/weights/"
    if not os.path.isdir(weights_directory):
        os.makedirs(weights_directory)

    start = datetime.datetime.now()
    configs['start'] = start
    configs['version'] = version

    with open(version_directory + '/configs.yml', 'w') as outfile:
        yaml.dump(configs, outfile, default_flow_style=False)

    # ========================= End of DevOps ==========================
    # ========================= Start of ML ==========================
    device = configs['device']
    print("Using", device)

    train_df = pd.read_csv(configs['train_df'])
    val_df = pd.read_csv(configs['val_df'])

    train_loader = dataloaders.get_train_dataloader(train_df, configs)
    val_loader = dataloaders.get_validation_dataloader(val_df, configs)

    model = SCANEncoder()
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=version_directory)

    if configs['print_model']:
        print(model)

    print("Starting training...")
    
    training_avg_losses = []
    for i in tqdm.trange(int(configs['max_epochs']), desc="Epoch"):
        training_losses = train(model, device, optim, criterion, train_loader,
                                writer, int(i+1))
        avg_loss = np.mean(training_losses)
        training_avg_losses.append(avg_loss)
        writer.add_scalar('Loss (Epoch)/Training', avg_loss, i)

        with torch.no_grad():
            # val_losses = validate(model, device, optim, criterion, train_loader,
            #                       writer, int(i+1))
            pass
        torch.save(model.state_dict(), weights_directory + "epoch_" + str(i) + ".pth")
    print("hi")
