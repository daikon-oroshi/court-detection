from typing import TypedDict, List, Tuple
from torch import Tensor
from PIL.Image import Image


class MarkedImage(TypedDict):
    image: Image
    org_size: Tuple[int, int]
    landmarks: List[Tuple[int, int]]


class MarkedImageTensor(TypedDict):
    image: Tensor
    org_size: Tuple[int, int]
    landmarks: Tensor
