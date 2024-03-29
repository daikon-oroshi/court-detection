from matplotlib import pyplot as plt
import torchvision as tv
from court_detection.utils import coord
from court_detection.data.data_set import BdcDataSet
from court_detection.data.transforms import (
    Resize, RandomErasing,
    HorizontalFlip, Grayscale,
    Squaring
)
from court_detection.env import env


class TestDrawLandmark(object):

    IMG_PATH = "../resources/image/court"
    LAND_PATH = env.LANDMARK_FILE

    def test_draw_landmark(self):
        size = (224, 224)
        transform = tv.transforms.Compose(
            [
                Squaring(),
                Resize(size),
                RandomErasing(),
                HorizontalFlip(),
                Grayscale()
            ]
        )
        ds = BdcDataSet(self.IMG_PATH, self.LAND_PATH, transform=transform)
        for i in range(0, min(9, len(ds))):

            train_data = ds[i]
            for pt_idx, pt in enumerate(train_data['landmarks']):
                img_coord = coord.to_img_coord(pt, size)
                plt.plot(
                    *img_coord,
                    marker='.',
                    c='r'
                )
                plt.text(
                    *img_coord,
                    str(pt_idx)
                )
            plt.imshow(train_data['image'])
            plt.show()


if __name__ == "__main__":
    test = TestDrawLandmark()
    test.test_draw_landmark()
