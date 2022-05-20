import torch
import librosa
import numpy as np
from typing import Dict, Any, Tuple, Union

import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

from dpe.dataset import create_train_val_dataloader
from dpe.model import PitchExtractor
from dpe.utils import read_config, pickle_binary


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
args = parser.parse_args()


if __name__ == '__main__':
    config = read_config(args.config)

    data_path = Path(config['data_dir'])
    batch_size = config['training']['batch_size']
    train_dataloader, val_dataloader = create_train_val_dataloader(
        data_path=data_path, batch_size=batch_size)

    model = PitchExtractor(spec_dim=513)

    for batch in train_dataloader:
        specs = batch['specs']
        out = model(specs)
        print(out['pitch'].size())
        print(out['logits'].size())