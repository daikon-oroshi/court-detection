import PIL
import random

from ...types.train_data import TrainData


class VerticalFlip:

    def __init__(
        self,
        p: float = 0.5
    ):
        self.p = p

    def mirror(self, x: float) -> float:
        return 1 - x

    def __call__(self, sample: TrainData) -> TrainData:

        if random.random() > self.p:
            return sample

        else:
            fliped = sample.image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            lmarks = list(map(
                lambda x: [self.mirror(x[0]), x[1]],
                sample.landmarks
            ))

            return TrainData(
                fliped,
                lmarks
            )
