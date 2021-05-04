import torchvision
from ...types.train_data import TrainData


class ToTensor:

    def __init__(self):
        self.transorm = torchvision.transforms.ToTensor()

    def __call__(self, sample: TrainData) -> TrainData:
        return TrainData(
            self.transform(sample.image),
            sample.landmarks
        )
