from typing import Tuple, List
import numbers
from torchvision.transforms import functional as F
from court_detection.data.types.marked_image import MarkedImage


class Squaring:

    def __init__(
        self,
        fill: int | Tuple = 0
    ):
        if not isinstance(fill, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate fill arg")
        self.fill = fill

    def __get_padding(
        self,
        img_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        w, h = img_size
        max_wh = max(w, h)
        h_padding = max_wh - w
        v_padding = max_wh - h

        left_pad = int(h_padding / 2)
        top_pad = int(v_padding / 2)
        right_pad = h_padding - left_pad
        bottom_pad = v_padding - top_pad

        return [left_pad, top_pad, right_pad, bottom_pad]

    def modify_landmarks_by_padding(
        self,
        landmarks: List[Tuple[int, int]],
        img_size: Tuple[int, int],
        padding: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int]]:

        def modify_landmark(
            x: int, y: int,
            img_size: Tuple[int, int],
            padding: Tuple[int, int, int, int]
        ) -> Tuple[int, int]:
            w, h = img_size
            left_pad, top_pad, right_pad, bottom_pad = padding
            return (
                (w * x + left_pad - right_pad) / (w + left_pad + right_pad),
                (h * y + bottom_pad - top_pad) / (h + bottom_pad + top_pad),
            )

        return [
            modify_landmark(x[0], x[1], img_size, padding)
            for x in landmarks
        ]

    def __call__(self, sample: MarkedImage) -> MarkedImage:

        padding = self.__get_padding(sample['image'].size)

        return {
            'image': F.pad(
                sample['image'],
                padding,
                self.fill
            ),
            'landmarks': self.modify_landmarks_by_padding(
                sample['landmarks'],
                sample['image'].size,
                padding
            )
        }
