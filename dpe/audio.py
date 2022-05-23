from pathlib import Path
from typing import Union

import librosa
import numpy as np
import torch


class AudioProcessor:

    def __init__(self,
                 sample_rate: int,
                 hop_length: int,
                 win_length: int,
                 n_fft: int,
                 pitch_min: int,
                 pitch_max: int) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max

    def load_wav(self, path: Union[Path, str]) -> torch.Tensor:
        wav, _ = librosa.load(str(path))
        wav = torch.tensor(wav)
        return wav

    def wav_to_spec(self, wav: torch.Tensor) -> torch.Tensor:
        spec = librosa.stft(
            y=wav.cpu().numpy(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length)
        spec = np.abs(spec)
        spec = torch.tensor(spec).float()
        return spec