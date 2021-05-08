import sys
import torch
from . import model


if __name__ == "__main__":
    img_path = sys.argv[1]
    land_path = sys.argv[2]
    save_path = sys.argv[3]

    net = model.Net(32)
    # model_ft = model.create_model(32)
    net.to('cpu')
    dataloaders, dataset_sizes = model.create_dataloader(img_path, land_path)
    learning_rate = 1e-4
    optimizer_ft = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer_ft, exp_lr_scheduler = model.create_optimizer(model_ft)
    criterion = torch.nn.MSELoss()
    # criterion = model.PointLoss()

    model_tr = model.train(
        'cpu',
        net,
        criterion,
        optimizer_ft,
        None,
        dataloaders,
        dataset_sizes,
        num_epochs=50
    )

    torch.save(model_tr.state_dict(), save_path)
