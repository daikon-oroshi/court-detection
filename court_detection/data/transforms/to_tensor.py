import torch
import torchvision
from ..types.marked_image import MarkedImage, MarkedImageTensor


class ToTensor:

    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()

    def flatten(self, landmarks):
        lms = []
        for lm in landmarks:
            lms.extend(lm)
        return lms

    def __call__(self, sample: MarkedImage) -> MarkedImageTensor:

        return {
            'image': self.transform(sample['image']),
            'landmarks': torch.tensor(
                self.flatten(sample['landmarks']),
                dtype=float
            )
        }
