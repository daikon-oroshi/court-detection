import sys
import torch
from . import model


if __name__ == "__main__":
    img_path = sys.argv[1]
    land_path = sys.argv[2]
    save_path = sys.argv[3]

    model_ft = model.create_model('cpu')
    dataloaders, dataset_sizes = model.create_dataloader(img_path, land_path)
    optimizer_ft, exp_lr_scheduler = model.create_optimizer(model_ft)
    criterion = torch.nn.MSELoss()

    model_tr = model.train(
        'cpu',
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=50
    )

    torch.save(model_tr.state_dict(), save_path)
