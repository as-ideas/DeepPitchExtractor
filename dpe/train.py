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


if __name__ == '__main__':
    config = read_config(args.config)

    data_path = Path(config['data_dir'])
    batch_size = config['training']['batch_size']
    train_dataloader, val_dataloader = create_train_val_dataloader(
        data_path=data_path, batch_size=batch_size)

    model = PitchExtractor(spec_dim=config['audio']['n_fft'] // 2 + 1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir='/tmp/pitch_log', comment='v1')
    step = 0
    ce_loss = torch.nn.CrossEntropyLoss()
    val_batches = [b for b in val_dataloader]

    for epoch in range(100):
        for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
            step += 1
            spec = batch['spec']
            out = model(spec)
            pitch_loss = F.l1_loss(out['pitch'].squeeze(1), batch['pitch'])
            bin_target = (batch['pitch'] != 0).long()
            bin_loss = ce_loss(out['logits'], bin_target)
            loss = pitch_loss + bin_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('pitch_loss', pitch_loss.item(), global_step=step)
            writer.add_scalar('bin_loss', bin_loss.item(), global_step=step)

            print(loss.item())

        val_batch = val_batches[0]
        with torch.no_grad():
            pred = model(val_batch['spec'])

        spec_len = val_batch['spec_len'][0]
        pitch_target_fig = plot_pitch(val_batch['pitch'][0, :spec_len].cpu().numpy())
        pitch_fig = plot_pitch(pred['pitch'][0, 0, :spec_len].cpu().numpy())

        bin_target = (val_batch['pitch'] != 0).long()
        bin_target_fig = plot_pitch(bin_target[0, :spec_len])
        bin_pred_probs = pred['logits'].softmax(1)
        bin_fig = plot_pitch(bin_pred_probs[0, 1, :spec_len])

        writer.add_figure('Pitch/target', pitch_target_fig, global_step=step)
        writer.add_figure('Pitch/pred', pitch_fig, global_step=step)
        writer.add_figure('Binary/target', bin_target_fig, global_step=step)
        writer.add_figure('Binary/pred', bin_fig, global_step=step)
