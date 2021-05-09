
FACTOR = 100


def to_ml(x, img_x):
    return ((x / img_x) - 0.5) * FACTOR


def to_img(x, img_x):
    return ((x / FACTOR) + 0.5) * img_x


def to_ml_coord(pt, img_size):
    return [
        to_ml(pt[0], img_size[0]),
        to_ml(pt[1], img_size[1]),
    ]


def to_img_coord(pt, img_size):
    return [
        to_img(pt[0], img_size[0]),
        to_img(pt[1], img_size[1])
    ]
