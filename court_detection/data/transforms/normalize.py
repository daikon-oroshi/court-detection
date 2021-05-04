import torchvision
from ...types.train_data import TrainData


class Normalize:

    def __init__(self, mean, std, inplace=False):
        self.transorm = torchvision.transforms.Normalize(mean, std, inplace)

    def __call__(self, sample: TrainData) -> TrainData:
        return TrainData(
            self.transform(sample.image),
            sample.landmarks
        )
