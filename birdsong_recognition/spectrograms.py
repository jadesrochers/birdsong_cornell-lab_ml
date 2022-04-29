import torch_librosa as tl
from torch import nn

## Making spectrograms given input data.
# This is created based on sample rate, window size, hop size for inputs.
class SpectrogramCreator():
    def __init__(self, sample_rate: int, window_size: int, hop_size: int,
    mel_bins: int=40, fmin: int=0, fmax: int=22050):
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
       self.hop_size = hop_size

       # Nyquist limit; the highest freq data can be extracted for.
       self.nyquist = sample_rate / 2
       # The last useful bin, which is the freq below the nyquist.
       self.max_useful_bin = int((self.nyquist * (self.window_size/self.sample_rate)) - 1)
       # The frequency for each bin: freq = bin * samp_rate / window_size
       self.freq_bins = [ i * self.sample_rate / self.window_size for i in range(self.max_useful_bin) ]
       self.fmin = fmin
       self.fmax = fmax
       if fmax > self.freq_bins[-1]:
           print('fmax set to: ', self.freq_bins[-1])
           self.fmax = self.freq_bins[-1]
       if fmin < self.freq_bins[0]:
           print('fmin set to: ', self.freq_bins[0])
           self.fmin = self.freq_bins[0]

       bins = 0
       for i in self.freq_bins:
          if i > self.fmin and i < self.fmax:
              bins+=1
       print('spectro fft bins: ', bins)
       self.fft_bins = bins
       print('mel bins: ', mel_bins)
       self.mel_bins = mel_bins
       # I can't calculate the mel bins yet, because I haven't 
       # figured it out. It will be less than the fft bins because
       # it alters it based on estimate human perception.

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
          n_fft=self.window_size,
          n_mels=self.mel_bins,
          fmin=self.fmin,
          fmax=self.fmax,
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
        if input.get_device() > -1:
            self.spectrogram_maker.cuda(input.get_device())
            self.logmel_extractor.cuda(input.get_device())
        spectrogram = self.spectrogram_maker(input)
        logmel_spectro = self.logmel_extractor(spectrogram)
        # frequencies = dft_frequencies(self.window_size, self.sample_rate, self.bins)
        # self.spectro_plot(logmel_spectro, frequencies)
        # if input.get_device() > -1:
        #     logmel_spectro.cuda(input.get_device())
        return logmel_spectro


    def spectro_plot(self, logmel_spectro, frequencies):
        # 4d, but just need the innermost arrays transposed.
        ten_seconds = math.ceil((self.sample_rate / self.hop_size) * 10)
        reformat = logmel_spectro.detach().numpy()[0, 0, 0:ten_seconds].transpose(1,0)
        # librosa.display.specshow(logmel_spectro)
        spec = plotex.imshow(reformat, y=frequencies, aspect='auto')
        spec.show()


