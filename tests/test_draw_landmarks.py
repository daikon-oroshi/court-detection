from matplotlib import pyplot as plt
import torchvision as tv
from court_detection import util
from court_detection.data.data_set import BdcDataSet
from court_detection.data.transforms import (
    Resize, RandomErasing,
    HorizontalFlip, Grayscale
)


class TestDrawLandmark(object):

    IMG_PATH = "../resources/image/court"
    LAND_PATH = "../resources/image/court/landmarks.json"

    def test_draw_landmark(self):
        size = (224, 224)
        transform = tv.transforms.Compose(
            [
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
                img_coord = util.to_img_coord(pt, size)
                plt.plot(
                    *img_coord,
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
