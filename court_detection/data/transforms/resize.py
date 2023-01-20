from typing import Tuple
import torchvision
from court_detection.data.types.marked_image import MarkedImage


class Resize:

    def __init__(self, output_size: Tuple[int, int]):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        self.transform = torchvision.transforms.Resize(output_size)

    def __call__(self, sample: MarkedImage) -> MarkedImage:
        return {
            'image': self.transform(sample['image']),
            'landmarks': sample['landmarks']
        }
