import sys
import torch
from . import train


if __name__ == "__main__":
    img_path = sys.argv[1]
    land_path = sys.argv[2]
    save_path = sys.argv[3]

    model_ft = train.create_model('cpu')
    dataloaders, dataset_sizes = train.create_dataloader(img_path, land_path)
    optimizer_ft, exp_lr_scheduler = train.create_optimizer(model_ft)
    criterion = torch.nn.MSELoss()

    model_tr = train.train(
        'cpu',
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        dataloaders,
        dataset_sizes
    )

    torch.save(model_tr.state_dict(), save_path)
