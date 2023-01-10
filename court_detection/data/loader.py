import torch
import torchvision
import os
from .data_set import BdcDataSet
from .transforms import (
    RandomErasing,
    Resize,
    HorizontalFlip,
    ToTensor,
    # Grayscale,
    Normalize
)
from court_detection.consts.train_phase import TrainPhase


def get_data_transforms(phase: TrainPhase) -> torchvision.transforms.Compose:
    size = (224, 224)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    re_scale = (0.02, 0.15)

    trans = [Resize(size)]
    if phase == TrainPhase.TRAIN:
        trans.extend([
            RandomErasing(scale=re_scale),
            HorizontalFlip()
        ])
    trans.extend([
        ToTensor(),
        # Grayscale(),
        # Normalize([0.5], [0.5]),
        Normalize(norm_mean, norm_std)
    ])

    return torchvision.transforms.Compose(trans)


def create_dataloader(img_paths: str, land_path: str, batch_size=4):
    phase = [
        TrainPhase.TRAIN,
        TrainPhase.VALIDATE
    ]

    data_transforms = {
        phase[0]: get_data_transforms(phase[0]),
        phase[1]: torchvision.transforms.Compose(phase[1]),
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