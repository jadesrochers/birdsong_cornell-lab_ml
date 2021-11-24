import pandas as pd
import numpy as np
import librosa
import torch
import librosa.display as display
import soundfile as sf
import torch_librosa as tl
import torch.nn as nn
import plotly.express as plotex
from pathlib import Path



# Examples of torch layers. Lay out how the model I am studying works in simple examples, then go from there.
# I really just need to see how they are coming up with full clip vs segment (sub-segments of the clip) predictions.

root_dir = Path.cwd()
raw_audio_dir = root_dir / 'train_audio'
resample_audio_dir =  root_dir / 'train_audio_resampled'
resample_audio_csv_file =  root_dir / 'train_resample.csv'
train_csv_file = root_dir / 'train.csv'

train_csv_df = pd.read_csv(train_csv_file)


## These might get pulled into model data loaders, or preprocess methods.
# I want to first make sure they work right however, so start with them broken out.

## Data loading

## Data Preprocess
# This is part of the PANNsCNN14Att model in the 'Intro to Sound Event Detection'
# module. 
# I want to first see what the output looks like for data using this preprocessing,
# then I can hook it up to the model and do all of that.
class DataPreprocess():
    def __init__(self, sample_rate: int, window_size: int, hop_size: int,
    mel_bins: int, fmin: int, fmax: int):
       window = 'hann'
       center = True
       pad_mode = 'reflect'
       ref = 1.0
       amin = 1e-9
       top_db = None
       # downsampled ratio. How can I know this in advance?
       self.interpolate_ratio = 32
       self.sample_rate = sample_rate
       self.window_size = window_size
       self.bins = mel_bins

       # Result should be the same as librosa.core.stft for producting a Spectrogram. 
       # Uses conv1d to do this, replicating the Stft from librosa.
       self.spectrogram_maker = tl.Spectrogram(
          n_fft = window_size,
          hop_length = hop_size,
          win_length = window_size,
          window = window,
          center = center,
          pad_mode = pad_mode,
          freeze_parameters=True
       )

       self.logmel_extractor = tl.LogmelFilterBank(
          sr = sample_rate,
          n_fft=window_size,
          n_mels=mel_bins,
          fmin=fmin,
          fmax=fmax,
          ref=ref,
          amin=amin,
          top_db=top_db,
          freeze_parameters=True
        )

       self.batch_normal2d = nn.BatchNorm2d(mel_bins)


    def preprocess(self, input, mixup_lambda=None):
        spectrogram = self.spectrogram_maker(input)
        logmel_spectro = self.logmel_extractor(spectrogram)
        frequencies = dft_frequencies(self.window_size, self.sample_rate, self.bins)
        spectro_plot(logmel_spectro, frequencies)
        return logmel_spectro


def spectro_plot(logmel_spectro, frequencies):
    import pdb; pdb.set_trace()
    # 4d, but just need the innermost arrays transposed.
    reformat = logmel_spectro.detach().numpy()[0, 0, 0:200].transpose(1,0)
    # librosa.display.specshow(logmel_spectro)
    spec = plotex.imshow(reformat, y=frequencies, aspect='auto')
    spec.show()

# The frequencies start at 0Hz (flat line defined by Fhat0)
# and go up to the nyquist frequency, which is the sample rate/2.
def dft_frequencies(sample_size, sample_rate, bins):
    freq_hz = []
    conversion_factor = sample_size / 2 / bins
    for f in range(bins):
        f_hat = f * conversion_factor
        freq_hz.append(f_hat * sample_rate / sample_size)
    return freq_hz


def time_series(xdata, ydata, xlab, ylab):
    df = pd.DataFrame()
    df[xlab] = xdata
    df[ylab] = ydata
    fig = plotex.line(df, x=xlab, y=ylab) 
    fig.show()

sample_rate = 16000

def explore_data(start=10, end=20):
    data_csv = pd.read_csv(resample_audio_csv_file)
    subset = data_csv.iloc[start:end, :]
    # I downsampled to 16000. The spectrogram parameters may need to be
    # adjusted, but this is a starting point.
    # The spectrogram might not include enough high frequencies for birds.
    preprocessor = DataPreprocess(sample_rate, 1024, 1024, 64, 50, 8000)
    for i, row in subset.iterrows():
        ebird_code = row.ebird_code
        resampled_filename = row.resampled_filename
        data_path = resample_audio_dir / ebird_code / resampled_filename
        sound_data = sf.read(data_path)
        x_data = list(range(160000))
        # time_series([x / sample_rate for x in x_data], sound_data[0][0:160000], 'Time(s)', 'Soundish')
        # The input for the spectrogram is just a tensor of the sound values.
        # It needed to be float() due to the torchlibrosa stuff, but was very 
        # confusing because the error seemed indicated needing a double.
        # Maybe something else multiplied by the sound was throwing the error.
        sound_sample = torch.from_numpy(sound_data[0][0:sample_rate*50]).float()
        unsqueeze = sound_sample.unsqueeze(0)
        rslt = preprocessor.preprocess(unsqueeze)

if __name__ == "__main__":
    explore_data()
