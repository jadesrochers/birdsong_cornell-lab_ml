import torch_librosa as tl
from torch import nn

## Making spectrograms given input data.
# This is created based on sample rate, window size, hop size for inputs.
class SpectrogramCreator():
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
       self.hop_size = hop_size

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

    # Calculate frequency stuff for FFT, based on:
    # https://stackoverflow.com/questions/4364823/how-do-i-obtain-the-frequencies-of-each-value-in-an-fft/4371627#4371627
    def dft_frequencies(sample_size, sample_rate, bins):
        freq_hz = []
        conversion_factor = sample_size / 2 / bins
        for f in range(bins):
            f_hat = f * conversion_factor
            freq_hz.append(f_hat * sample_rate / sample_size)
        return freq_hz

    # Get a spectrogram using the torch librosa functions.
    # If I want to do augmentations, create another Class/Module to do that.
    def spectro_from_data(self, input, mixup_lambda=None):
        spectrogram = self.spectrogram_maker(input)
        logmel_spectro = self.logmel_extractor(spectrogram)
        # frequencies = dft_frequencies(self.window_size, self.sample_rate, self.bins)
        # self.spectro_plot(logmel_spectro, frequencies)
        return logmel_spectro


    def spectro_plot(self, logmel_spectro, frequencies):
        # 4d, but just need the innermost arrays transposed.
        ten_seconds = math.ceil((self.sample_rate / self.hop_size) * 10)
        reformat = logmel_spectro.detach().numpy()[0, 0, 0:ten_seconds].transpose(1,0)
        # librosa.display.specshow(logmel_spectro)
        spec = plotex.imshow(reformat, y=frequencies, aspect='auto')
        spec.show()


