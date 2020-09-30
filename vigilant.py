import torch
import torch.nn as nn
import torchvision
import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(
        model,
        device,
        optimizer: torch.optim.Optimizer,
        criterion,
        dataloader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        epoch: int
):
    model.train()
    
    losses = []
    for batch_index, (img, label) in enumerate(dataloader):
        img = img.to(device)
        labels = label.to(device)

        optimizer.zero_grad()

        output = model(img)
        loss = criterion(output, labels)
        loss.backward()
        
        optimizer.step()

        # print("Training loss: ", loss.item(), flush=True)
        writer.add_scalar('Loss (steps)/Training Epoch ' + str(epoch), loss.item(), batch_index)    # b index * 2, b_index * 3
        losses.append(loss.item())

    return losses


def validate(
        model,
        device,
        optimizer: torch.optim.Optimizer,
        criterion,
        dataloader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        epoch: int
):
    model.eval()

    losses = []

    for batch_index, (img, label) in enumerate(dataloader):
        img = img.to(device)
        labels = label.to(device)

        optimizer.zero_grad()

        output = model(img)
        loss = criterion(output, labels)
        loss.backward()

        # print("Training loss: ", loss.item(), flush=True)
        writer.add_scalar('Loss (steps)/Validation Epoch ' + str(epoch), loss.item(), batch_index)
        losses.append(loss.item())

    return losses
