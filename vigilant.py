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
        optimizer,
        criterion,
        scheduler,
        dataloader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        epoch: int,
        config: dict,
        debug: bool = False
):
    model.train()
    
    losses = []
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

        output = model(img)
        loss = criterion(output, labels)
        loss.backward()
        
        optimizer.step()
        
        predictions = torch.argmax(output, dim=1).cpu().numpy()
        acer, apcer, npcer = metrics.get_metrics(predictions, labels.cpu())
        acc = accuracy_score(labels.cpu(), predictions)

        # print("Training loss: ", loss.item(), flush=True)
        writer.add_scalar('Loss/Training', loss.item(), epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (training)/acer', acer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (training)/apcer', apcer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (training)/npcer', npcer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (training)/accuracy', acc, epoch * len(dataloader) + batch_index)

        losses.append(loss.item())
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

    return losses, metrics_dict


def validate(
        model,
        device,
        optimizer,
        criterion,
        dataloader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        epoch: int,
        config: dict,
        debug: bool = False
):
    model.eval()

    losses = []
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

        output = model(img)
        loss = criterion(output, labels)

        predictions = torch.argmax(output, dim=1).cpu().numpy()
        acer, apcer, npcer = metrics.get_metrics(predictions, labels.cpu())
        acc = accuracy_score(labels.cpu(), predictions)

        # print("Training loss: ", loss.item(), flush=True)
        writer.add_scalar('Loss/Validation', loss.item(), epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (validation)/acer', acer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (validation)/apcer', apcer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (validation)/npcer', npcer, epoch * len(dataloader) + batch_index)
        writer.add_scalar('Metrics (validation)/accuracy', acc, epoch * len(dataloader) + batch_index)

        losses.append(loss.item())
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
