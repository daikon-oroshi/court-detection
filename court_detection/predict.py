import sys
import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt
from . import model
from court_detection.data.transforms import (
    Resize,
    ToTensor,
    Normalize
)

if __name__ == "__main__":
    model_path = sys.argv[1]
    img_path = sys.argv[2]

    net = model.create_model('cpu')
    net.load_state_dict(torch.load(model_path))

    net.eval()

    dataloaders, _ = model.create_dataloader(img_path, None)
    dataloaders = iter(dataloaders['train'])

    with torch.no_grad():
        to_pil = tv.transforms.ToPILImage()
        for sample in dataloaders:

            lms = net(sample['image'].to('cpu'))

            for d in range(lms.size()[0]):

                lm = lms[d, :].flatten().tolist()
                img = sample['image'][d, :, :, :]
                for i in range(0, len(lm), 2):
                    plt.plot(
                        224 * lm[i], 224 * lm[i + 1],
                        marker='x', color="red"
                    )
                plt.imshow(to_pil(img).convert("RGB"))
                plt.show()
