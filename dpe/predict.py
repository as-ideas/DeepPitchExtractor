import torch
from dpe.model import PitchModel
from dpe.audio import AudioProcessor
from dpe.utils import denormalize_pitch

if __name__ == '__main__':

    path = '/Users/cschaefe/datasets/bild_snippets_cleaned/Snippets/r_0696_014.wav'

    checkpoint = torch.load('/Users/cschaefe/workspace/DeepPitchExtractor/checkpoints/latest_model.pt',
                            map_location=torch.device('cpu'))

    config = checkpoint['config']
    audio = AudioProcessor(**config['audio'])
    wav = audio.load_wav(path)
    spec = audio.wav_to_spec(wav).unsqueeze(0)
    model = PitchModel(in_channels=config['audio']['n_fft'] // 2 + 1,
                       out_channels=config['model']['out_channels'] + 2,
                       conv_channels=config['model']['conv_channels'],
                       dropout=config['model']['dropout'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    pred = model(spec)
    pitch = torch.argmax(pred, dim=1)
    pitch = denormalize_pitch(pitch=pitch,
                              pmin=config['audio']['pitch_min'],
                              pmax=config['audio']['pitch_max'],
                              n_channels=config['model']['out_channels'])

    print(pitch)