from typing import Tuple

FACTOR = 1


def _to_ml(x: float, img_x: int) -> float:
    """
    0 <= x <= img_x -> -0.5 <= x <= 0.5
    """
    return ((x / img_x) - 0.5) * FACTOR


def _to_img(x: float, img_x: int) -> float:
    """
    -0.5 <= x <= 0.5 -> 0 <= x <= img_x
    """
    return ((x / FACTOR) + 0.5) * img_x


def to_ml_coord(
    pt: Tuple[float, float],
    img_size: Tuple[int, int]
) -> Tuple[float, float]:
    return (
        _to_ml(pt[0], img_size[0]),
        _to_ml(pt[1], img_size[1])
    )


def to_img_coord(
    pt: Tuple[float, float],
    img_size: Tuple[int, int]
) -> Tuple[float, float]:
    return [
        _to_img(pt[0], img_size[0]),
        _to_img(pt[1], img_size[1])
    ]
