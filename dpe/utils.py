import torch
import pickle
import yaml
from pathlib import Path
from typing import Union, List, Any, Dict, Tuple


def get_files(path: Union[str, Path], extension='.wav') -> List[Path]:
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))


def pickle_binary(data: object, file: Union[str, Path]) -> None:
    with open(str(file), 'wb') as f:
        pickle.dump(data, f)


def unpickle_binary(file: Union[str, Path]) -> Any:
    with open(str(file), 'rb') as f:
        return pickle.load(f)


def read_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, 'w+', encoding='utf-8') as stream:
        yaml.dump(config, stream, default_flow_style=False)


def normalize_pitch(pitch: torch.Tensor,
                    pmin: int,
                    pmax: int,
                    n_channels: int) -> torch.Tensor:
    pitch = torch.clone(pitch)
    valid_inds = torch.logical_and(pmin <= pitch, pmax >= pitch)
    pitch = (pitch - pmin) / (pmax - pmin) * n_channels + 1
    pitch = torch.round(pitch).long()
    pitch[~valid_inds] = 0
    return pitch


def denormalize_pitch(pitch: torch.Tensor,
                      pmin: int,
                      pmax: int,
                      n_channels: int) -> torch.Tensor:
    pitch = torch.clone(pitch)
    valid_inds = pitch > 0
    pitch = (pitch - 1) * (pmax - pmin) / n_channels + pmin
    pitch[~valid_inds] = 0
    pitch = pitch.float()
    return pitch