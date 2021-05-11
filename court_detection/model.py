import time
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import math

from .data.data_set import BdcDataSet
from .data.transforms import (
    RandomErasing,
    Resize,
    VerticalFlip,
    ToTensor,
    Normalize
)


class Net(nn.Module):

    def __init__(self, output_size, pretrained=True):
        super(Net, self).__init__()

        resnet = torchvision.models.resnet152(pretrained=pretrained)
        # resnet = torchvision.models.resnet18(pretrained=pretrained)
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


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def train(
    device,
    net,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=25
):
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = None

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode

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
                    outputs = net(inputs)
                    loss = criterion(outputs.float(), labels.float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and not math.isnan(epoch_loss) and \
                    (best_loss is None or best_loss > epoch_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())

            if phase == 'val' and best_loss is not None:
                print('BEST Loss: {:.4f}'.format(best_loss))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net


def create_dataloader(img_paths: str, land_path: str, batch_size=4):
    phase = ['train', 'val']
    size = (224, 224)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transforms = {
        phase[0]: torchvision.transforms.Compose([
            Resize(size),
            # RandomErasing(scale=(0.02, 0.15)),
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
