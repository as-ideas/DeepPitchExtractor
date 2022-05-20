from pathlib import Path
from typing import Dict, List, Union, Tuple

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from dpe.utils import unpickle_binary


class PitchDataset(Dataset):

    def __init__(self,
                 item_ids: List[str],
                 spec_path: Path,
                 pitch_path: Path):
        self._item_ids = item_ids
        self._spec_path = spec_path
        self._pitch_path = pitch_path

    def __getitem__(self, index: int) -> Dict[str, Union[str, torch.Tensor]]:
        item_id = self._item_ids[index]
        spec = torch.load(self._spec_path / f'{item_id}.pt')
        pitch = torch.load(self._pitch_path / f'{item_id}.pt')
        return {'spec': spec, 'pitch': pitch, 'item_id': item_id}

    def __len__(self):
        return len(self._item_ids)


def collate_dataset(batch: List[dict]) -> Dict[str, torch.Tensor]:
    specs = [b['spec'] for b in batch]
    specs = pad_sequence(specs, batch_first=True, padding_value=0)
    pitches = [b['pitch'] for b in batch]
    pitches = pad_sequence(pitches, batch_first=True, padding_value=0)
    item_ids = [b['item_id'] for b in batch]
    item_ids = torch.tensor(item_ids)
    return {'specs': specs, 'pitches': pitches, 'item_ids': item_ids}


def create_train_val_dataloader(data_path: Path, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_data = unpickle_binary(data_path/'train_dataset.pkl')
    val_data = unpickle_binary(data_path/'val_dataset.pkl')
    train_ids, train_lens = zip(*train_data)
    val_ids, val_lens = zip(*val_data)
    train_dataset = PitchDataset(item_ids=train_ids,
                                 spec_path=data_path / 'specs',
                                 pitch_path=data_path / 'pitches')
    val_dataset = PitchDataset(item_ids=val_ids,
                               spec_path=data_path / 'specs',
                               pitch_path=data_path / 'pitches')
    train_loader = DataLoader(train_dataset,
                              collate_fn=lambda batch: collate_dataset(batch),
                              batch_size=batch_size,
                              sampler=None,
                              num_workers=0,
                              pin_memory=True)

    return train_loader, train_loader


if __name__ == '__main__':
    train_dataloader, val_dataloader = create_train_val_dataloader(Path('/Users/cschaefe/workspace/DeepPitchExtractor/data'), batch_size=2)

    for batch in train_dataloader:
        print(batch)