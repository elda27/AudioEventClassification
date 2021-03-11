from pydantic import BaseModel

import json
from pathlib import Path
from enum import Enum
from typing import List, Dict, Optional

from libs.model import NetworkType
from libs.misc.json_codec import PickleEncoder, PickleDecoder


class Model(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        json_loads = lambda *args, **kwargs: json.loads(*args, cls=PickleEncoder, **kwargs)
        json_dumps = lambda *args, **kwargs: json.dumps(*args, cls=PickleDecoder, **kwargs)


class InputType(Enum):
    Audio = "audio"
    Spectrum = "spectrum"


class PreprocessProps(Model):
    enable_minmax_normalization: bool = True


class TrainingProps(Model):
    learning_rate: float = 1e-4
    exponential_decay: Optional[float] = None
    warmup_step: Optional[int] = None


class Props(Model):
    fold_index = 0
    task = 'home'
    data_dir: Path() = './data'

    training_props: TrainingProps

    model_dir: Path = './model'

    model_type: NetworkType = NetworkType.CNN

    @property
    def input_type(self) -> InputType:
        if self.model_type in (NetworkType.BigBird, NetworkType.Reformer):
            return InputType.Audio
        else:
            return InputType.Spectrum
