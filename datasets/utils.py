import torchvision
import matplotlib.pyplot as plt


def imshow(batch, title=None):
    """Imshow for Tensor."""
    images = torchvision.utils.make_grid(batch)
    images = images.detach().cpu().numpy().transpose((1, 2, 0))
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def construct_grid(batch):
    images = torchvision.utils.make_grid(batch)
    images = images.detach().cpu().numpy()
    return images
