from typing import Dict, Tuple
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import ResNet152_Weights
from court_detection.consts.train_phase import TrainPhase

from .data.data_set import BdcDataSet
from .data.transforms import (
    RandomErasing,
    Resize,
    HorizontalFlip,
    ToTensor,
    Grayscale,
    Normalize
)


class Net(nn.Module):

    def __init__(self, output_size, pretrained=True, grayscale=False):
        super(Net, self).__init__()

        weights = ResNet152_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet152(
            weights=weights
        )

        if grayscale:
            resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = resnet.fc.in_features
        self.resnet_base = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(num_ftrs, output_size)

    def forward(self, x):
        h = self.resnet_base(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)

        return h


class RMSELoss(nn.Module):

    def __init__(self, pt_dim=2):
        super().__init__()
        self.pt_dim = pt_dim

    def forward(self, output, target):
        sp_out = torch.split(output, self.pt_dim, dim=1)
        sp_tar = torch.split(target, self.pt_dim, dim=1)

        # root取らないので普通のmse_lossと同じ
        pt_wise_loss = [
            self.pt_loss(o.float(), t.float())
            for o, t in zip(sp_out, sp_tar)
        ]
        loss = torch.stack(pt_wise_loss, dim=1)
        loss = torch.mean(loss)

        return loss

    def pt_loss(self, o, t):
        l2 = torch.sqrt(torch.sum(
            torch.pow(torch.sub(o, t), 2),
            dim=1))
        return l2


def create_dataloader(img_paths: str, land_path: str, batch_size=4):
    phase = [
        TrainPhase.TRAIN,
        TrainPhase.VALIDATE
    ]
    size = (224, 224)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transforms = {
        phase[0]: torchvision.transforms.Compose([
            Resize(size),
            RandomErasing(scale=(0.02, 0.15)),
            HorizontalFlip(),
            ToTensor(),
            # Grayscale(),
            # Normalize([0.5], [0.5]),
            Normalize(norm_mean, norm_std)
        ]),
        phase[1]: torchvision.transforms.Compose([
            Resize(size),
            ToTensor(),
            # Grayscale(),
            # Normalize([0.5], [0.5])
            Normalize(norm_mean, norm_std)
        ]),
    }

    image_datasets = {
        x: BdcDataSet(
            os.path.join(img_paths, x),
            land_path,
            data_transforms[x]
        ) for x in phase
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size,
            shuffle=True, num_workers=1
        ) for x in phase
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in phase}

    return dataloaders, dataset_sizes


def create_optimizer(model_ft):
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)

    return optimizer_ft, exp_lr_scheduler


def save_state(path: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        path
    )


def load_state(
    path: str
) -> Tuple[int, Dict, Dict]:
    loaded_obj = torch.load(path)
    return loaded_obj["epoch"], \
        loaded_obj["model"], \
        loaded_obj["optimizer"]
