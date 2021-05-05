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

    size = (224, 224)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform = tv.transforms.Compose(
        [
            Resize(size),
            ToTensor(),
            Normalize(norm_mean, norm_std)
        ]
    )

    img = Image.open(img_path).convert('RGB')
    img.load()
    img_size = img.size

    img_trans = transform({'image': img, 'landmarks': []})['image']
    img_trans = img_trans[:3, :, :].unsqueeze(0)

    with torch.no_grad():
        lms = net(img_trans.to('cpu')).flatten().tolist()
    print(lms)

    for i in range(0, len(lms), 2):
        plt.plot(
            img_size[0] * lms[i], img_size[1] * lms[i + 1],
            marker='x', color="red"
        )
    plt.imshow(img)
    plt.show()
