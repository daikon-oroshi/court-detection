import type as t
import torchvision

from ..types.train_data import TrainData


class Resize:

    def __init__(self, output_size: t.Tuple[int, int]):
        assert isinstance(output_size, tuple)
        self.output_size = output_size
        self.transform = torchvision.transforms.Resize(output_size)

    def __call__(self, sample: TrainData) -> TrainData:

        return TrainData(
            self.transform(sample.image),
            sample.landmarks
        )
