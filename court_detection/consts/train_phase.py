from enum import Enum


class TrainPhase(str, Enum):
    TRAIN = "train"
    VALIDATE = "val"
    TEST = "test"
