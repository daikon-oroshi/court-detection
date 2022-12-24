import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt
from court_detection import model, util
from court_detection.data.transforms import (
    Resize,
    ToTensor,
)
from court_detection.env import env
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path', type=str,
    default=str(Path(env.MODELS_DIR) / Path(env.DEFAULT_MODEL_FILE_NAME)),
    help='model file path'
)
parser.add_argument(
    '--img_path',
    type=str,
    required=True,
    help='image path'
)

args = parser.parse_args()


if __name__ == "__main__":
    model_path = args.model_path
    img_path = args.img_path

    net = model.Net(32, grayscale=False)
    net.to('cpu')
    _, model_state, _ = model.load_state(model_path)
    net.load_state_dict(model_state)

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
