import typing as t
import PIL
import random


class VerticalFlip:

    def __init__(
        self,
        p: float = 0.5
    ):
        self.p = p

    def mirror(self, x: float) -> float:
        return -x

    def flip_landmarks(self, landmarks):
        mirror_lmarks = list(map(
            lambda x: [self.mirror(x[0]), x[1]],
            landmarks
        ))

        lmarks = []

        # 2 <-> 0
        lmarks.append(mirror_lmarks[2])
        lmarks.append(mirror_lmarks[1])
        lmarks.append(mirror_lmarks[0])
        # 3 <-> 5
        lmarks.append(mirror_lmarks[5])
        lmarks.append(mirror_lmarks[4])
        lmarks.append(mirror_lmarks[3])
        # 6 <-> 7
        lmarks.append(mirror_lmarks[7])
        lmarks.append(mirror_lmarks[6])
        # 8 <-> 10
        lmarks.append(mirror_lmarks[10])
        lmarks.append(mirror_lmarks[9])
        lmarks.append(mirror_lmarks[8])
        # 11 <-> 13
        lmarks.append(mirror_lmarks[13])
        lmarks.append(mirror_lmarks[12])
        lmarks.append(mirror_lmarks[11])
        # 14 <-> 15
        lmarks.append(mirror_lmarks[15])
        lmarks.append(mirror_lmarks[14])

        return lmarks

    def __call__(self, sample) -> t.Tuple:

        if random.random() > self.p:
            return sample

        else:
            fliped = sample['image'].transpose(PIL.Image.FLIP_LEFT_RIGHT)
            lmarks = self.flip_landmarks(sample['landmarks'])

            return {
                'image': fliped,
                'landmarks': lmarks
            }
