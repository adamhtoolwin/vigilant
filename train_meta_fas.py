import argparse
import yaml
import datetime
import os
import glob
import tqdm

import utils
from dataset import datasets
from models.scan import SCAN, Vigilant, SCANEncoder
from models.resnet import ResNet18Classifier
from vigilant import *

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Config file path.")
    parser.add_argument("-ch", "--checkpoint", required=False, help="Checkpoint file path.", default=None)
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

    debug = False

    start = datetime.datetime.now()
    configs['start'] = start
    configs['version'] = version
    configs['debug'] = debug

    with open(version_directory + '/configs.yml', 'w') as outfile:
        yaml.dump(configs, outfile, default_flow_style=False)

    # ========================= End of DevOps ==========================
    # ========================= Start of ML ==========================
    device = configs['gpu']
    print("Using", device)
    print("Version ", version)
    print("Debug: ", debug)

    train_df = pd.read_csv(configs['train_df'])
    val_df = pd.read_csv(configs['val_df'])

    train_loader = datasets.get_train_dataloader(train_df, configs)
    val_loader = datasets.get_validation_dataloader(val_df, configs)

    model = ResNet18Classifier(pretrained=configs['pretrained'])

    try:
        model.load_state_dict(configs['checkpoint'])
    except KeyError:
        pass

    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=configs['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=configs['milestones'], gamma=configs['gamma']
    )

    writer = SummaryWriter(log_dir=version_directory)

    if configs['print_model']:
        print(model)

    print("====================")
    print("Starting training...")

    training_avg_losses = []
    val_avg_losses = []

    for i in range(int(configs['max_epochs'])):
        training_losses, training_metrics = train(model, device, optim, criterion, scheduler, train_loader,
                                                  writer, int(i + 1), configs, debug)

        train_avg_loss = np.mean(training_losses)
        train_avg_metric = {
            'acer': np.mean(training_metrics['acer']),
            'apcer': np.mean(training_metrics['apcer']),
            'npcer': np.mean(training_metrics['npcer']),
            'acc': np.mean(training_metrics['acc'])
        }

        training_avg_losses.append(train_avg_loss)

        writer.add_scalar('Loss (epoch)/Training average loss', train_avg_loss, i)
        writer.add_scalar('Training metrics (epoch)/Training average acer', train_avg_metric['acer'], i)
        writer.add_scalar('Training metrics (epoch)/Training average apcer', train_avg_metric['apcer'], i)
        writer.add_scalar('Training metrics (epoch)/Training average npcer', train_avg_metric['npcer'], i)
        writer.add_scalar('Training metrics (epoch)/Training average accuracy', train_avg_metric['acc'], i)

        with torch.no_grad():
            val_losses, val_metrics = validate(model, device, optim, criterion, val_loader,
                                               writer, int(i + 1), configs, debug)
            val_avg_loss = np.mean(val_losses)
            val_avg_metric = {
                'acer': np.mean(val_metrics['acer']),
                'apcer': np.mean(val_metrics['apcer']),
                'npcer': np.mean(val_metrics['npcer']),
                'acc': np.mean(val_metrics['acc'])
            }

            val_avg_losses.append(val_avg_loss)

        writer.add_scalar('Loss (epoch)/Validation average loss', val_avg_loss, i)
        writer.add_scalar('Val metrics (epoch)/Validation average acer', val_avg_metric['acer'], i)
        writer.add_scalar('Val metrics (epoch)/Validation average apcer', val_avg_metric['apcer'], i)
        writer.add_scalar('Val metrics (epoch)/Validation average npcer', val_avg_metric['npcer'], i)
        writer.add_scalar('Val metrics (epoch)/Validation average accuracy', val_avg_metric['acc'], i)

        torch.save(model.state_dict(), weights_directory + "epoch_" + str(i) + ".pth")

        # scheduler.step(epoch=None)
        # current_lr = scheduler.get_last_lr()[0]
        # writer.add_scalar('Learning Rate', current_lr, i)

    plt.plot(training_avg_losses, label="Training average loss")
    plt.plot(val_avg_losses, label="Validation average loss")
    plt.title("Overall Loss curve")
    plt.legend()
    plt.grid()
    plt.savefig(version_directory + "/losses.png")
    writer.close()
