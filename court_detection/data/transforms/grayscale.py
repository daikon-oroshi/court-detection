import torchvision
from court_detection.data.types.marked_image import MarkedImage


class Grayscale:

    def __init__(self):
        self.transform = torchvision.transforms.Grayscale()

    def __call__(self, sample: MarkedImage) -> MarkedImage:
        return {
            'image': self.transform(sample['image']),
            'landmarks': sample['landmarks']
        }
