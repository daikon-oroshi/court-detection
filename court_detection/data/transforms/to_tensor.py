import torchvision
import numpy as np


class ToTensor:

    def __init__(self):
        self.transform = torchvision.transforms.ToTensor()

    def __call__(self, sample):

        return {
            'image': self.transform(sample['image']),
            'landmarks': self.transform(np.array(sample['landmarks']))
        }
