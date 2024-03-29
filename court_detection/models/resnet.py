import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet152_Weights


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
