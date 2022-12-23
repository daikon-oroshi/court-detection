from typing import Optional, Dict, Any
import pathlib
from pydantic import (
    BaseSettings, validator
)
from court_detection.consts.train_phase import TrainPhase


class Env(BaseSettings):

    LANDMARK_FILE: str

    @validator("LANDMARK_FILE", pre=True)
    def exists_landmark_file(cls, v: str) -> str:
        _path = pathlib.Path(v)
        assert _path.exists(), f"landmark file: not exist on {v}"
        assert not _path.is_dir(), f"landmark file: {v} is directory"
        return v

    DATA_DIR: str
    TRAIN_DATA_DIR: Optional[str]
    VALID_DATA_DIR: Optional[str]
    TEST_DATA_DIR: Optional[str]

    @validator("TRAIN_DATA_DIR", pre=True)
    def assemble_train_data_dir(
        cls,
        v: Optional[str],
        values: Dict[str, Any]
    ) -> str:
        train_data_dir = v
        if train_data_dir is None:
            train_data_dir = values.get("DATA_DIR") + TrainPhase.TRAIN.value

        _path = pathlib.Path(train_data_dir)
        assert _path.exists(), \
            f"train data dir: not exist on {train_data_dir}"
        assert _path.is_dir(), \
            f"train data dir: {train_data_dir} is not directory"

        return train_data_dir

    @validator("VALID_DATA_DIR", pre=True)
    def assemble_valid_data_dir(
        cls,
        v: Optional[str],
        values: Dict[str, Any]
    ) -> str:
        valid_data_dir = v
        if valid_data_dir is None:
            valid_data_dir = values.get("DATA_DIR") + TrainPhase.VALIDATE.value

        _path = pathlib.Path(valid_data_dir)
        assert _path.exists(), \
            f"valid data dir: not exist on {valid_data_dir}"
        assert _path.is_dir(), \
            f"valid data dir: {valid_data_dir} is not directory"
        return valid_data_dir

    TEST_DATA_DIR: Optional[str]

    @validator("TEST_DATA_DIR", pre=True)
    def is_dir_test_data_dir(
        cls,
        v: Optional[str],
        values: Dict[str, Any]
    ) -> str:
        if v is None:
            return v
        test_data_dir = values.get("DATA_DIR") + TrainPhase.VALIDATE.test

        _path = pathlib.Path(test_data_dir)
        assert _path.exists(), \
            f"test data dir: not exist on {test_data_dir}"
        assert _path.is_dir(), \
            f"test data dir: {test_data_dir} is not directory"
        return test_data_dir

    MODELS_DIR: str

    @validator("MODELS_DIR", pre=True)
    def exists_models_dir(cls, v: str) -> str:
        _path = pathlib.Path(v)
        assert _path.exists(), f"model dir: not exist on {v}"
        assert _path.is_dir(), f"model dir: {v} is not directory"
        return v

    DEFAULT_MODEL_FILE_NAME: str = "model.pth"

    class Config:
        case_sensitive = True
        env_file = '.env'


env = Env()
