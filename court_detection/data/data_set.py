import typing as t
import json
import re
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from ..utils import coord


class BdcDataSet(Dataset):

    def __init__(self, img_path: str, land_path: str, transform=None):
        super().__init__()

        self.transform = transform

        self.image_files = [
            p for p in Path(img_path).glob("**/*")
            if re.search('/*.(jpg|png)', str(p))
        ]
        if land_path is not None:
            with open(land_path) as lm:
                landmarks = json.load(lm)
            self.landmarks = self.__normalize_landmarks(landmarks)
        else:
            self.landmarks = {}

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        p = self.image_files[idx]
        with Image.open(str(p)).convert('RGB') as img:
            img.load()
            lmarks = self.landmarks.get(p.name, [])

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
            with Image.open(str(p)).convert('RGB') as img:
                img.load()
                norm_lands[p.name] = list(map(
                    lambda x: coord.to_ml_coord(x, img.size),
                    lmarks
                ))

        return norm_lands
