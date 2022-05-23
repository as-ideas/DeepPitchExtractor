import torch
import librosa

from dpe.model import PitchModel
from dpe.utils import denormalize_pitch

if __name__ == '__main__':

    path = '/Users/cschaefe/datasets/bild_snippets_cleaned/Snippets/r_0686_007.wav'

    checkpoint = torch.load('checkpoints/latest_model.pt')

    config = checkpoint['config']

    wav, _ = librosa.load(str(path))
    spec = librosa.stft(y=wav,
                        n_fft=config['audio']['n_fft'],
                        hop_length=config['audio']['hop_length'],
                        win_length=config['audio']['win_length'])

    model = PitchModel(in_channels=config['audio']['n_fft'] // 2 + 1,
                       out_channels=config['model']['out_channels'] + 2,
                       conv_channels=config['model']['conv_channels'],
                       dropout=config['model']['dropout'])

    spec = torch.tensor(spec).unsqueeze(0)
    pred = model(spec)
    pitch = torch.argmax(pred, dim=1)
    pitch = denormalize_pitch(pitch=pitch,
                              pmin=config['audio']['pitch_min'],
                              pmax=config['audio']['pitch_max'],
                              n_channels=config['model']['out_channels'])

    print(pred)