import typing as t
import torchvision


class Resize:

    def __init__(self, output_size: t.Tuple[int, int]):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        self.transform = torchvision.transforms.Resize(output_size)

    def __call__(self, sample):
        return {
            'image': self.transform(sample['image']),
            'landmarks': sample['landmarks']
        }
