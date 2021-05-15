import typing as t
import torchvision


class Grayscale:

    def __init__(self):
        self.transform = torchvision.transforms.Grayscale()

    def __call__(self, sample):
        return {
            'image': self.transform(sample['image']),
            'landmarks': sample['landmarks']
        }
