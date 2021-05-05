import typing as t
import json
import re
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from ..types.train_data import TrainData


class BdcDataSet(Dataset):

    def __init__(self, img_path: str, land_path: str, transform=None):
        super().__init__()

        self.transform = transform

        self.image_files = [
            p for p in Path(img_path).glob("**/*")
            if re.search('/*.(jpg|png)', str(p))
        ]
        with open(land_path) as lm:
            landmarks = json.load(lm)

        self.landmarks = self.__normalize_landmarks(landmarks)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx) -> TrainData:
        p = self.image_files[idx]
        img = Image.open(str(p)).convert('RGB')
        img.load()
        lmarks = self.landmarks[p.name]

        sample = {
            'image': img,
            'landmarks': lmarks
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __normalize_landmarks(self, landmarks) -> t.Dict:
        norm_lands = {}

        for p in self.image_files:
            lmarks = landmarks[p.name]
            img = Image.open(str(p)).convert('RGB')
            img.load()
            (width, height) = img.size
            norm_lands[p.name] = list(map(
                lambda x: [x[0] / width, x[1] / height],
                lmarks
            ))

        return norm_lands
