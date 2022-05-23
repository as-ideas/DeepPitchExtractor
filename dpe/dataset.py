from pathlib import Path
from typing import Dict, List, Union, Tuple
from torch.utils.data.sampler import Sampler
import torch
import random
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
        return {'spec': spec, 'pitch': pitch, 'item_id': item_id, 'spec_len': spec.size(-1)}

    def __len__(self):
        return len(self._item_ids)



class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)


def collate_dataset(batch: List[dict]) -> Dict[str, torch.Tensor]:
    pitches = [b['pitch'] for b in batch]
    pitches = pad_sequence(pitches, batch_first=True, padding_value=0)
    specs = [b['spec'].transpose(0, 1) for b in batch]
    specs = pad_sequence(specs, batch_first=True, padding_value=0)
    specs = specs.transpose(1, 2)
    spec_len = [b['spec_len'] for b in batch]
    item_ids = [b['item_id'] for b in batch]
    return {'spec': specs, 'pitch': pitches, 'item_id': item_ids, 'spec_len': spec_len}


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
                              sampler=BinnedLengthSampler(train_lens, batch_size, batch_size*3),
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            collate_fn=lambda batch: collate_dataset(batch),
                            batch_size=batch_size,
                            sampler=BinnedLengthSampler(val_lens, batch_size, batch_size*3),
                            num_workers=0,
                            pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    train_dataloader, val_dataloader = create_train_val_dataloader(Path('/Users/cschaefe/workspace/DeepPitchExtractor/data'), batch_size=2)

    for batch in train_dataloader:
        print(batch)