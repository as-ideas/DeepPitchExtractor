import torch
import librosa
import matplotlib as mpl
from matplotlib.figure import Figure
import numpy as np
mpl.use('agg')  # Use non-interactive backend by default
import matplotlib.pyplot as plt

from typing import Dict, Any, Tuple, Union
import torch.nn.functional as F
import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

from dpe.dataset import create_train_val_dataloader
from dpe.model import PitchExtractor
from dpe.utils import read_config, pickle_binary
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
args = parser.parse_args()


def plot_pitch(pitch: np.array, color='gray') -> Figure:
    fig = plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(pitch, color=color)
    return fig


def normalize_pitch(pitch: torch.Tensor,
                    pmin: int,
                    pmax: int,
                    n_channels: int) -> torch.Tensor:
    valid_inds = torch.logical_and(pmin <= pitch, pmax >= pitch)
    pitch[valid_inds] = (pitch[valid_inds] - pmin) / pmax * n_channels
    pitch[~valid_inds] = 0
    pitch = pitch.long()
    return pitch


if __name__ == '__main__':
    config = read_config(args.config)

    data_path = Path(config['paths']['data_dir'])
    cp_path = Path(config['paths']['checkpoint_dir'])
    cp_path.mkdir(parents=True, exist_ok=True)
    batch_size = config['training']['batch_size']
    train_dataloader, val_dataloader = create_train_val_dataloader(
        data_path=data_path, batch_size=batch_size)

    model = PitchExtractor(in_channels=config['audio']['n_fft'] // 2 + 1,
                           out_channels=config['model']['out_channels'],
                           conv_channels=config['model']['conv_channels'],
                           dropout=config['model']['dropout'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir='/tmp/pitch_log', comment='v1')
    step = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    val_batches = sorted([b for b in val_dataloader], key=lambda x: x['spec_len'][0])
    pmin, pmax = config['audio']['pitch_min'], config['audio']['pitch_max']
    out_channels = config['model']['out_channels']

    for epoch in range(config['training']['n_epochs']):
        for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
            step += 1
            spec = batch['spec']
            logits = model(spec).squeeze(1)
            pitch_target = normalize_pitch(batch['pitch'],
                                           pmin=pmin, pmax=pmax, n_channels=out_channels)
            loss = ce_loss(logits, pitch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=step)

        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'step': step}, cp_path / 'latest_model.pt')

        val_loss, val_batch, logits = 0, None, None
        for val_batch in val_batches:
            pitch_target = normalize_pitch(val_batch['pitch'],
                                           pmin=pmin, pmax=pmax, n_channels=out_channels)
            with torch.no_grad():
                logits = model(val_batch['spec'])
                loss = ce_loss(logits, pitch_target)
            val_loss += loss

        spec_len = val_batch['spec_len'][0]
        pitch_target = normalize_pitch(val_batch['pitch'],
                                       pmin=pmin, pmax=pmax, n_channels=out_channels)
        pitch_pred = torch.argmax(logits, dim=1)
        pitch_pred_nonzeros = torch.argmax(logits[:, 1:, :], dim=1) + 1
        pitch_target_fig = plot_pitch(pitch_target[0, :spec_len].cpu().numpy())
        pitch_pred_fig = plot_pitch(pitch_pred[0, :spec_len].cpu().numpy())
        pitch_pred_nonzero_fig = plot_pitch(pitch_pred[0, :spec_len].cpu().numpy())
        writer.add_figure('Pitch/target', pitch_target_fig, global_step=step)
        writer.add_figure('Pitch/pred', pitch_pred_fig, global_step=step)
        writer.add_figure('Pitch/pred_nonzero', pitch_pred_nonzero_fig, global_step=step)
        writer.add_scalar('Loss/val', float(val_loss) / len(val_batches), global_step=step)
