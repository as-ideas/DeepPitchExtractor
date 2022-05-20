from random import Random

import pyworld as pw
import torch
import librosa
import numpy as np
from typing import Dict, Any, Tuple, Union

import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

from dpe.utils import read_config, pickle_binary


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError('%r must be an integer greater than 0' % num)
    return n


class Preprocessor:

    def __init__(self,
                 data_dir: Path,
                 sample_rate: int,
                 hop_length: int,
                 win_length: int,
                 n_fft: int) -> None:
        self._data_dir = data_dir
        self._spec_dir = self._data_dir / 'specs'
        self._pitch_dir = self._data_dir / 'pitches'
        self._spec_dir.mkdir(parents=True, exist_ok=True)
        self._pitch_dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._hop_length = hop_length
        self._win_length = win_length
        self._n_fft = n_fft
        self._hann_window = torch.hann_window(self._win_length)

        pass

    def __call__(self, path: Path) -> Union[Tuple[str, int], None]:
        try:
            item_id = path.stem
            wav, _ = librosa.load(str(path))
            spec = librosa.stft(
                y=wav,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                win_length=self._win_length)
            spec = np.abs(spec)
            spec = torch.tensor(spec).float()
            pitch, _ = pw.dio(wav.astype(np.float64), self._sample_rate,
                              frame_period=self._hop_length / self._sample_rate * 1000)
            pitch = torch.tensor(pitch).float()
            torch.save(spec, self._spec_dir / f'{item_id}.pt')
            torch.save(pitch, self._pitch_dir / f'{item_id}.pt')
            spec_len = spec.shape[-1]
            return item_id, spec_len
        except BaseException as e:
            print(e)
            return None


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path')
parser.add_argument('--num_workers', '-w', metavar='N', type=valid_n_workers, default=cpu_count()-1, help='The number of worker threads to use for preprocessing')
parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
args = parser.parse_args()


if __name__ == '__main__':
    config = read_config(args.config)
    data_dir = Path(config['data_dir'])
    wav_files = list(Path(args.path).glob('**/*.wav'))
    n_workers = max(1, args.num_workers)
    pool = Pool(processes=n_workers)
    preprocessor = Preprocessor(data_dir=data_dir, **config['audio'])
    dataset = []
    for data_point in tqdm.tqdm(pool.imap_unordered(preprocessor, wav_files), total=len(wav_files)):
        if data_point is not None:
            dataset.append(data_point)
    Random(42).shuffle(dataset)
    num_val = config['training']['num_val']
    val_dataset = dataset[:num_val]
    train_dataset = dataset[num_val:]
    pickle_binary(train_dataset, data_dir / 'train_dataset.pkl')
    pickle_binary(val_dataset, data_dir / 'val_dataset.pkl')