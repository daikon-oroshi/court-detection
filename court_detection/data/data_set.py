import typing as t
import os
import json
import re
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from ..types.train_data import TrainData


class BdcDataSet(Dataset):

    def __init__(self, dir_path: str, transform=None):
        super().__init__()

        self.dir_path = dir_path
        self.transform = transform

        self.image_paths = [
            p for p in Path(self.dir_path).glob("**/*")
            if re.search('/*.(jpg|png)', str(p))
        ]
        with open(os.path.join(self.dir_path, "landmarks.json")) as lm:
            landmarks = json.load(lm)

        self.landmarks = self.__normalize_landmarks(landmarks)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> TrainData:
        p = self.image_paths[idx]
        img = Image.open(str(p))
        lmarks = self.landmarks[p.name]
        sample = TrainData(img, lmarks)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __normalize_landmarks(self, landmarks) -> t.Dict:
        norm_lands = {}

        for p in self.image_paths:
            lmarks = landmarks[p.name]
            img = Image.open(str(p))
            (width, height) = img.size
            norm_lands[p.name] = list(map(
                lambda x: [x[0] / width, x[1] / height],
                lmarks
            ))

        return norm_lands
