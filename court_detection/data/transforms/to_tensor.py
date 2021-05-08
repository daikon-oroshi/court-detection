import torch
import torchvision


class ToTensor:

    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()

    def flatten(self, landmarks):
        lms = []
        for lm in landmarks:
            lms.extend(lm)
        return lms

    def __call__(self, sample):

        return {
            'image': self.transform(sample['image']),
            'landmarks': torch.tensor(
                self.flatten(sample['landmarks']),
                dtype=float
            )
        }
