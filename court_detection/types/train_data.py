import types as t
from dataclasses import dataclass
from PIL import Image


@dataclass
class TrainData:
    image: Image
    landmarks: t.List[t.Tuple[float, float]]
