import sys
import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt
from . import model
from . import util
from .data.transforms import (
    Resize,
    ToTensor,
)


if __name__ == "__main__":
    model_path = sys.argv[1]
    img_path = sys.argv[2]

    net = model.Net(32, grayscale=False)
    net.to('cpu')
    net.load_state_dict(torch.load(model_path))

    net.eval()

    size = (224, 224)
    transform = tv.transforms.Compose([
        Resize(size),
        ToTensor()
    ])

    img = Image.open(img_path).convert('RGB')
    img.load()

    trans = transform({'image': img, 'landmarks': []})

    with torch.no_grad():

        lms = net(trans['image'].unsqueeze(0).to('cpu'))

        lm = lms[0, :].flatten().tolist()
        for i in range(0, len(lm), 2):
            img_coord = util.to_img_coord(lm[i:i+2], (224, 224))
            plt.plot(
                *img_coord,
                marker='.', color="red"
            )
            plt.text(
                *img_coord,
                str(int(i/2))
            )
        plt.imshow(img.resize(size))
        plt.show()
