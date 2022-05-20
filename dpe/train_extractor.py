from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PitchDataset(Dataset):

    def __init__(self,
                 item_ids: List[str],
                 spec_path: Path,
                 pitch_path: Path):
        self._item_ids = item_ids
        self._spec_path = spec_path
        self._pitch_path = pitch_path

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item_id = self._item_ids[index]
        spec = np.load(str(self._spec_path / f'{item_id}.pt'))
        pitch = np.load(str(self._pitch_path / f'{item_id}.pt'))
        spec = torch.tensor(spec).float()
        pitch = torch.tensor(pitch).float()
        return {'spec': spec, 'pitch': pitch}