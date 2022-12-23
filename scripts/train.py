import torch
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

args = parser.parse_args()


if __name__ == "__main__":
    img_data_path: str = env.DATA_DIR
    land_path: str = env.LANDMARK_FILE
    save_path = Path(env.MODELS_DIR) / Path(args.save_model_name)
    assert save_path.parent.is_dir(), \
        f"save_path: {save_path} is invalid path."

    net = model.Net(32, grayscale=False)
    # model_ft = model.create_model(32)
    net.to('cpu')
    dataloaders, dataset_sizes = model.create_dataloader(
        img_data_path, land_path, 8)
    learning_rate = 1e-4
    optimizer_ft = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer_ft, exp_lr_scheduler = model.create_optimizer(model_ft)
    criterion = torch.nn.MSELoss()

    scheduler = None
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer_ft, milestones=[20, 40], gamma=0.1)

    model_tr = train(
        'cpu',
        net,
        criterion,
        optimizer_ft,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=50
    )

    torch.save(model_tr.state_dict(), save_path)
