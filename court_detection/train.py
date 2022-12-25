from typing import Dict, Any
import torch
import time
import copy
import math
from court_detection import model
from court_detection.env import env
from court_detection.consts.train_phase import TrainPhase
from torch.utils.tensorboard import SummaryWriter


def train(
    device: str,
    net: torch.nn.Module,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dataloaders: Dict[TrainPhase, Any],
    dataset_sizes: Dict[TrainPhase, int],
    save_path: str,
    start_epoch: int = 0,
    num_epochs: int = 25,
    save_steps: int = 10
):
    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = None
    writer = SummaryWriter(log_dir=env.LOG_DIR)

    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in [TrainPhase.TRAIN, TrainPhase.VALIDATE]:
            if phase == TrainPhase.TRAIN:
                net.train()
            else:
                net.eval()

            running_loss = 0.0

            # Iterate over data.
            for td in dataloaders[phase]:
                inputs = td['image'].to(device)
                labels = td['landmarks'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == TrainPhase.TRAIN):
                    outputs = net(inputs)
                    loss = criterion(outputs.float(), labels.float())

                    if phase == TrainPhase.TRAIN:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)

            if phase == TrainPhase.TRAIN:
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
                if scheduler is not None:
                    scheduler.step()

            # deep copy the model
            if phase == TrainPhase.VALIDATE:
                if not math.isnan(epoch_loss) and \
                        (best_loss is None or best_loss > epoch_loss):
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(net.state_dict())

                if best_loss is not None:
                    writer.add_scalar("Loss/BEST", best_loss, epoch)

        if epoch % save_steps == 0:
            model.save_state(
                save_path,
                epoch,
                net,
                optimizer
            )

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    writer.close()

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net
