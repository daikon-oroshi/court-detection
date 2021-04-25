import typing as t
import torchvision

from ...types.train_data import TrainData


class RandomErasing:

    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0, inplace=False
    ):
        self.transform = torchvision.transforms.RandomErasing(
            p=p, scale=scale, ratio=ratio, value=value, inplace=inplace
        )

    def __call__(self, sample: TrainData) -> TrainData:

        to_tensor = torchvision.transforms.ToTensor()
        to_pil = torchvision.transforms.ToPILImage()

        transed_tensor = self.transform(to_tensor(sample.image))
        transed_pilim = to_pil(transed_tensor).convert("RGB")

        return TrainData(
            transed_pilim,
            sample.landmarks
        )
