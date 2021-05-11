import torch
from court_detection.model import RMELoss
import torch.nn.functional as F


class TestPointLoss(object):

    def test_point_loss(self):
        point_loss = RMELoss()

        output = torch.tensor(
            [
             [0, 1, 2, 3, 4, 5],
             [0, 1, 2, 3, 4, 5],
             [8, 9, 10, 11, 12, 13],
             [8, 9, 10, 11, 12, 13],
            ]
        )
        target = torch.tensor(
            [
             [0, 0, 0, 0, 0, 0],
             [16, 17, 18, 19, 20, 21],
             [16, 17, 18, 19, 20, 21],
             [24, 25, 26, 27, 28, 29],
             # [24, 25, 26, 27, 28, 29, 30, 31],
            ]
        )
        print(output.size())
        loss = point_loss(output, target)
        print(loss)
        print(F.mse_loss(output.float(), target.float()))


if __name__ == "__main__":
    test = TestPointLoss()
    test.test_point_loss()
