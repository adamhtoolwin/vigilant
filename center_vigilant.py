import torch
import cv2
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.utils import construct_grid
import metrics
from sklearn.metrics import accuracy_score


def train(
        model,
        device,
        optimizer: torch.optim.Optimizer,
        criterion,
        dataloader: torch.utils.data.DataLoader,
        center_optimizer: torch.optim.Optimizer,
        center_criterion,
        writer: SummaryWriter,
        epoch: int,
        config: dict,
        debug: bool = False,
        alpha: int = 1
):
    model.train()
    
    cross_losses = []
    center_losses = []
    metrics_dict = {
        'acer': [],
        'apcer': [],
        'npcer': [],
        'acc': []
    }

    pbar = tqdm(dataloader)
    pbar.set_description("Epoch %d training" % epoch)
    for batch_index, (img, label) in enumerate(pbar):
        img = img.to(device)
        labels = label.to(device)

        optimizer.zero_grad()
        center_optimizer.zero_grad()

        features, outputs = model(img)

        cross_entropy_loss = criterion(outputs, labels)
        center_loss = criterion(features, labels)
        center_loss *= alpha

        loss = cross_entropy_loss + center_loss

        loss.backward()

        for param in center_criterion.parameters():
            param.grad.data *= (1. / alpha)

        optimizer.step()
        center_optimizer.step()
        
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        acer, apcer, npcer = metrics.get_metrics(predictions, labels.cpu())
        acc = accuracy_score(labels.cpu(), predictions)

        # print("Training loss: ", loss.item(), flush=True)
        writer.add_scalar('Loss/Training', cross_entropy_loss.item(), epoch * len(dataloader) + batch_index)
        writer.add_scalar('Center Loss/Training', center_loss.item(), epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (training)/acer', acer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (training)/apcer', apcer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (training)/npcer', npcer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (training)/accuracy', acc, epoch * len(dataloader) + batch_index)

        cross_losses.append(cross_entropy_loss.item())
        center_losses.append(center_loss.item())
        metrics_dict['acer'].append(acer)
        metrics_dict['apcer'].append(apcer)
        metrics_dict['npcer'].append(npcer)
        metrics_dict['acc'].append(acc)

        if debug:
            print("\nRan training sanity check...")
            break

    if config['cue_log_every_epoch']:
        # get random sample from dataloader
        imgs, labels = next(iter(dataloader))
        #
        # imgs = imgs.to(device)
        # output = model(imgs)
        # imgs = imgs.cpu()
        #
        # predictions = list(torch.argmax(output, dim=1).cpu().numpy())
        # predictions_string = " ".join(predictions)

        images_grid = construct_grid(imgs)
        # cues_grid = construct_grid(labels[-1])

        # writer.add_image("Training/Cues", cues_grid, epoch)
        writer.add_image("Training/Images", images_grid, epoch)

    return cross_losses, metrics_dict


def validate(
        model,
        device,
        optimizer: torch.optim.Optimizer,
        criterion,
        dataloader: torch.utils.data.DataLoader,
        center_optimizer: torch.optim.Optimizer,
        center_criterion,
        writer: SummaryWriter,
        epoch: int,
        config: dict,
        debug: bool = False,
        alpha: int = 1
):
    model.eval()

    cross_losses = []
    center_losses = []

    metrics_dict = {
        'acer': [],
        'apcer': [],
        'npcer': [],
        'acc': []
    }

    pbar = tqdm(dataloader)
    pbar.set_description("Epoch %d validation" % epoch)
    for batch_index, (img, label) in enumerate(pbar):
        img = img.to(device)
        labels = label.to(device)

        optimizer.zero_grad()
        center_optimizer.zero_grad()

        features, outputs = model(img)
        cross_entropy_loss = criterion(outputs, labels)
        center_loss = center_criterion(features, labels)

        loss = center_loss * alpha + cross_entropy_loss

        # for param in center_criterion.parameters():
        #     param.grad.data *= (1. / alpha)

        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        acer, apcer, npcer = metrics.get_metrics(predictions, labels.cpu())
        acc = accuracy_score(labels.cpu(), predictions)

        # print("Training loss: ", loss.item(), flush=True)
        writer.add_scalar('Loss/Validation', cross_entropy_loss.item(), epoch * len(dataloader) + batch_index)
        writer.add_scalar('Center Loss/Validation', center_loss.item(), epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (validation)/acer', acer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (validation)/apcer', apcer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (validation)/npcer', npcer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (validation)/accuracy', acc, epoch * len(dataloader) + batch_index)

        cross_losses.append(cross_entropy_loss.item())
        center_losses.append(center_loss.item())
        metrics_dict['acer'].append(acer)
        metrics_dict['apcer'].append(apcer)
        metrics_dict['npcer'].append(npcer)
        metrics_dict['acc'].append(acc)

        if debug:
            print("\nRan validation sanity check...")
            break

    if config['cue_log_every_epoch']:
        # get random sample from dataloader
        imgs, labels = next(iter(dataloader))

        images_grid = construct_grid(imgs)
        # cues_grid = construct_grid(labels[-1])

        # writer.add_image("Training/Cues", cues_grid, epoch)
        writer.add_image("Validation/Images", images_grid, epoch)

    return losses, metrics_dict


def evaluate():
    pass
