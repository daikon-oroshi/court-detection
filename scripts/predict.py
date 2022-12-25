import torch
import torchvision as tv
from matplotlib import pyplot as plt
from court_detection import model, util
from court_detection.env import env
from court_detection.consts.train_phase import TrainPhase
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path', type=str,
    default=str(Path(env.MODELS_DIR) / Path(env.DEFAULT_MODEL_FILE_NAME)),
    help='model file path'
)

args = parser.parse_args()


if __name__ == "__main__":
    model_path = args.model_path
    img_path = env.DATA_DIR

    net = model.Net(32, grayscale=False, pretrained=False)
    net.to('cpu')
    _, model_state, _ = model.load_state(model_path)
    net.load_state_dict(model_state)

    net.eval()

    land_path = env.LANDMARK_FILE
    dataloaders, _ = model.create_dataloader(img_path, land_path, 1)

    with torch.no_grad():
        to_pil = tv.transforms.ToPILImage()
        for sample in iter(dataloaders[TrainPhase.VALIDATE]):

            lms = net(sample['image'].to('cpu'))
            print(sample['landmarks'])
            print(lms)

            for d in range(lms.size()[0]):

                lm = lms[d, :].flatten().tolist()
                img = sample['image'][d, :, :, :]
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
                plt.imshow(to_pil(img).convert("RGB"))
                plt.show()
