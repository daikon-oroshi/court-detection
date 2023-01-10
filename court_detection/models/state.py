from typing import Dict, Tuple
import torch
import torch.nn as nn


# TODO: modelクラス毎に実装

def save_state(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        path
    )


def load_state(
    path: str,
    device: str = "cpu"
) -> Tuple[int, Dict, Dict]:
    loaded_obj = torch.load(path, map_location=device)
    return loaded_obj["epoch"], \
        loaded_obj["model"], \
        loaded_obj["optimizer"]
