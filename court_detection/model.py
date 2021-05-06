import time
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .data.data_set import BdcDataSet
from .data.transforms import (
    RandomErasing,
    Resize,
    VerticalFlip,
    ToTensor,
    Normalize
)


def create_model(device: str):
    model_ft = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 32)

    model_ft = model_ft.to(device)

    return model_ft


def train(
    device,
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=25
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = None

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for td in dataloaders[phase]:
                inputs = td['image'].to(device)
                labels = td['landmarks'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # tensor(max, max_indices)なのでpredは0,1のラベル
                    loss = criterion(outputs.float(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and \
                    (best_loss is None or best_loss > epoch_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def create_dataloader(img_paths: str, land_path: str):
    phase = ['train', 'val']
    size = (224, 224)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transforms = {
        phase[0]: torchvision.transforms.Compose([
            Resize(size),
            RandomErasing(scale=(0.02, 0.15)),
            VerticalFlip(),
            ToTensor(),
            Normalize(norm_mean, norm_std)
        ]),
        phase[1]: torchvision.transforms.Compose([
            Resize(size),
            ToTensor(),
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
            image_datasets[x], batch_size=4,
            shuffle=True, num_workers=1
        ) for x in phase
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in phase}

    return dataloaders, dataset_sizes


def create_optimizer(model_ft):
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=3, gamma=0.1)

    return optimizer_ft, exp_lr_scheduler
