import torchvision
from court_detection.data.types.marked_image import MarkedImageTensor


class Normalize:

    def __init__(self, mean, std, inplace=False):
        self.transform = torchvision.transforms.Normalize(mean, std, inplace)

    def __call__(self, sample: MarkedImageTensor) -> MarkedImageTensor:
        return {
            'image': self.transform(sample['image']),
            'landmarks': sample['landmarks']
        }
