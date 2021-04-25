import type as t
import os
import json
import re
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class BdcDataSet(Dataset):

    def __init__(self, dir_path, transform=None):
        super().__init__()

        self.dir_path = dir_path
        self.transform = transform

        self.image_paths = [
            p for p in Path(self.dir_path).glob("**/*")
            if re.search('/*.(jpg|png)', str(p))]
        ]
        with open(os.path.join(self.dir_path, "landmark.json")) as lm:
            landmarks = json.load(lm)

        self.landmarks = self.__normalize_landmarks(landmarks)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # TODO: 整理
        image_path = self.image_paths[idx]

        img = Image.open(str(p))
        landmarks = self.landmarks[p.name]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __normalize_landmarks(self, landmarks) -> t.List:
        norm_lands = {}

        for p in self.image_paths:
            lmarks = landmarks[p.name]
            img = Image.open(str(p))
            (width, height) = img.size
            norm_lands[p.name] = map(
                lambda x: [x[0] / width, x[1] / height],
                lmarks
            )

        return norm_lands
