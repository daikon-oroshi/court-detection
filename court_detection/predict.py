import sys
import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt
from . import model
from . import util
from court_detection.data.transforms import (
    Resize,
    ToTensor,
    Normalize
)

if __name__ == "__main__":
    model_path = sys.argv[1]
    img_path = sys.argv[2]

    net = model.Net(32)
    net.to('cpu')
    net.load_state_dict(torch.load(model_path))

    net.eval()

    dataloaders, _ = model.create_dataloader(img_path, None)
    dataloaders = iter(dataloaders['val'])

    with torch.no_grad():
        to_pil = tv.transforms.ToPILImage()
        for sample in dataloaders:

            lms = net(sample['image'].to('cpu'))

            for d in range(lms.size()[0]):

                lm = lms[d, :].flatten().tolist()
                img = sample['image'][d, :, :, :]
                for i in range(0, len(lm), 2):
                    plt.plot(
                        *util.to_img_coord(lm[i:i+1], (224, 224)),
                        marker='x', color="red"
                    )
                plt.imshow(to_pil(img).convert("RGB"))
                plt.show()
