import torch
from torch import nn
from torchvision import models

from models.decoder import Decoder
from models.resnet import ResNet18Classifier, ResNet18Encoder


class SCAN(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.backbone = ResNet18Encoder()
        self.decoder = Decoder()
        self.clf = ResNet18Classifier(dropout=dropout)

    def forward(self, x):
        x = x.float()
        outs = self.backbone(x)
        outs = self.decoder(outs)

        s = x + outs[-1]
        clf_out = self.clf(s)

        return outs, clf_out


class SCANEncoder(nn.Module):
    def __init__(self, dropout: float = 0.5, num_classes: int = 2):
        super().__init__()
        self.backbone = ResNet18Encoder(pretrained=False)
        self.fc = nn.Linear(
            in_features=25088, out_features=num_classes
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.float()
        outs = self.backbone(x)

        s = outs[-1]
        s = torch.flatten(s, 1)
        s = self.drop(s)
        s = self.fc(s)

        return s


class Vigilant(nn.Module):
    def __init__(self, dropout: float = 0.5, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.fc = nn.Linear(
            in_features=25088, out_features=num_classes
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.backbone(x)
        z = torch.flatten(x, 1)
        z = self.drop(z)
        output = self.fc(z)

        return output
