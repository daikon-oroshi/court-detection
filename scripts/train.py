import sys
import torch
from court_detection import model
import time
import copy
import math


def train(
    device,
    net,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=25
):
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = None

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for td in dataloaders[phase]:
                inputs = td['image'].to(device)
                labels = td['landmarks'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs.float(), labels.float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'train':
                print('LR: {}'.format(optimizer.param_groups[0]['lr']))
                if scheduler is not None:
                    scheduler.step()

            # deep copy the model
            if phase == 'val':
                if not math.isnan(epoch_loss) and \
                        (best_loss is None or best_loss > epoch_loss):
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(net.state_dict())

                if best_loss is not None:
                    print('BEST Loss: {:.4f}'.format(best_loss))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net


if __name__ == "__main__":
    img_path = sys.argv[1]
    land_path = sys.argv[2]
    save_path = sys.argv[3]

    net = model.Net(32, grayscale=False)
    # model_ft = model.create_model(32)
    net.to('cpu')
    dataloaders, dataset_sizes = model.create_dataloader(
        img_path, land_path, 8)
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
