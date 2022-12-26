from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from court_detection import model
from court_detection.train import train
from court_detection.env import env
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_model_name', type=str,
    default=env.DEFAULT_MODEL_FILE_NAME,
    help='name of model file'
)
parser.add_argument(
    '--device', type=str,
    default='cpu',
    help='device'
)
parser.add_argument(
    '--epochs', type=int,
    default=100,
    help='number of epochs'
)

args = parser.parse_args()


LEARNING_RATE = 1e-5


def load_model(
    model_path: Optional[str]
) -> Tuple[int, nn.Module, Optional[optim.Optimizer]]:

    not_exists_model = model_path is None or not Path(model_path).exists()
    net = model.Net(32, grayscale=False, pretrained=not_exists_model)
    net.to(args.device)

    optimizer_ft = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    if not_exists_model:
        return 1, net, optimizer_ft

    epoch, model_state, optim_state = model.load_state(
        model_path,
        device=args.device
    )
    net.load_state_dict(model_state)
    optimizer_ft.load_state_dict(optim_state)

    return epoch, net, optimizer_ft


if __name__ == "__main__":
    img_data_path: str = env.DATA_DIR
    land_path: str = env.LANDMARK_FILE
    model_path = Path(env.MODELS_DIR) / Path(args.save_model_name)

    assert model_path.parent.is_dir(), \
        f"save_path: {model_path} is invalid path."

    epoch, net, optimizer_ft = load_model(model_path)
    dataloaders, dataset_sizes = model.create_dataloader(
        img_data_path, land_path, 8
    )
    criterion = torch.nn.MSELoss()
    scheduler = None

    model_tr = train(
        args.device,
        net,
        criterion,
        optimizer_ft,
        scheduler,
        dataloaders,
        dataset_sizes,
        model_path,
        start_epoch=epoch,
        num_epochs=args.epochs,
    )
