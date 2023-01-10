import torch
import torch.nn as nn
import timm


class Net(nn.Module):

    def __init__(self, output_size, pretrained=True, grayscale=False):
        super(Net, self).__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=output_size
        )
        # TODO: grayscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
