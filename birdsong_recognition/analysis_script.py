import pandas as pd
import numpy as np
import librosa
import librosa.display as display
import soundfile as sf
import plotly.express as plotex
import math
import torch
import torch_librosa as tl
import torch.nn as nn
import random
import re
from ast import literal_eval

# Includes the DataLoader
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold


root_dir = Path.cwd()
raw_audio_dir = root_dir / 'train_audio'
resample_audio_dir =  root_dir / 'train_audio_resampled'
resample_audio_csv_file =  root_dir / 'train_resample.csv'
train_csv_file = root_dir / 'train.csv'

train_csv_df = pd.read_csv(train_csv_file)


## These might get pulled into model data loaders, or preprocess methods.
# I want to first make sure they work right however, so start with them broken out.

## Data loading
# This is directly pulled from one of the Birdsound notebooks. 
# Figure out how it works and see what you need to do to get it working with your set.
## Problems:
# 1. This only pulls a single data chunk from the start of the data.
#    I really need to pull more than that, yet at the same time if the analysis
#    window is 5 seconds, then I need to train on 5 second windows.
# 1. Solution: Pull a configurable number of chunks out of the data. You can do this 
#    Randomly or from the start. I think random non-overlapping would be good.
# This will require a more complex output where there is a list of data, list of 
# labels. The current notebook I am looking at also uses primary/secondary
# labels and does data augmentation, so I am going to try and dip into those 
# areas, with augmentation specifically for the training data.  
class AudioDataset(Dataset):
    def __init__(
            self,
            file_list: pd.DataFrame,
            category_codes: Dict,
            period,
            spectrogram_maker,
            is_training=True,
            waveform_transforms=None):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.category_codes = category_codes
        self.waveform_transforms = waveform_transforms
        self.spectrogram_maker = spectrogram_maker
        self.is_training = is_training
        # This is the length of epoch?
        # Is there any way to get anything except the first epoch?
        self.period = period

    def __len__(self):
        return len(self.file_list)

    #def get_label_data(self, idx):
    #    return self.file_list[idx]

    # This appears to do onehot encoding. I would rather avoid this
    # if possible, I thought generally torch can do without it.
    def convert_labels_to_coded(self, num_images, labels):
        coded_labels = np.zeros((num_images, len(self.bird_code)))
        for index, temp_label in enumerate(labels):
            label_index = self.bird_code[temp_label]
            coded_labels[:,label_index] = 1
        return torch.from_numpy(coded_labels).float()

    # Get training samples for a file. Will tacke the number of samples
    # specified with the length of each (epoch_size) in seconds.
    def get_training_sample(self, data, sr, samples=5, epoch_size=5):
       complete_epochs = int(len(data) / sr // epoch_size)
       sampled_epochs = random.sample(list(range(complete_epochs)), samples)
       sample_idxs = [(epoch * sr, (epoch + 1) * sr) for epoch in sampled_epochs]
       spectros = []
       for (start, end) in sample_idxs:
           epoch_data = data[start:end]
           epoch_tensor = torch.from_numpy(epoch_data).float()
           # Unsqueeze because the spectrogram maker uses a conv1d and
           # can accept 1 or more channels, so need channel nested even if
           # there is only 1.
           epoch_unsqueeze = epoch_tensor.unsqueeze(0)
           spectros.append(self.spectrogram_maker.spectro_from_data(epoch_unsqueeze))
       return spectros

    # When predicting, do I just get the whole set of data?
    # That seems like a logical way to do it, but I will have to determine
    # how the model fits in.
    # 1. It seems like an Event Detection model should be able to predict
    #    on each epoch (5s) and then summarize to the whole length by 
    #    averaging those, no matter how many there are
    # 2. Maybe having a length that evenly divides into the epochs will
    #    be important however, so perhaps I should do that here.
    def get_prediction_data(self, data, sr, epoch_limit=100):
       complete_epochs = int(len(data) / sr // epoch_size)
       analyze_epochs = epoch_limit if complete_epochs > epoch_limit else complete_epochs
       prediction_epochs = list(range(analyze_epochs))
       spectro_idxs = [(epoch * sr, (epoch + 1) * sr) for epoch in prediction_epochs]
       spectros = []
       for (start, end) in sample_idxs:
           epoch_data = data[start:end]
           epoch_tensor = torch.from_numpy(epoch_data).float()
           # See training data note about unsqueeze
           epoch_unsqueeze = epoch_tensor.unsqueeze(0)
           spectros.append(self.spectrogram_maker.spectro_from_data(epoch_unsqueeze))
      return spectros


    # Get the code, filename. Load the data, sample a set of epochs from the 
    def __getitem__(self, idx: int):
        row = self.file_list.iloc[idx, :]
        y, sr = sf.read(row.resampled_full_path)
        # This will ultimately return a dict with the list of spectrograms, codes,
        # and some other stuff.
        if(self.is_training){
            spectros = self.get_training_sample(y, sr, samples=5, epoch_size=5)
        }else{
            spectros = self.get_prediction_data(y, sr, epoch_limit=100)
        }
        import pdb; pdb.set_trace()
        # I did not think I needed one-hot vector labels for torch;
        # add these back in if it is not working.
        # all_labels = self.convert_labels_to_coded(len(spectros), row.all_labels)
        # primary_labels = self.convert_labels_to_coded(len(spectros), row.primary_label)
        import pdb; pdb.set_trace()

        # TODO:
        # Add in Augmentations that will do various stretch, compress, noise
        # on the data to make it work better for the unknown data to be predicted on.
        #if self.waveform_transforms:
        #    y = self.waveform_transforms(y)
        #else:
        #    len_y = len(y)
        #    effective_length = sr * self.period
        #    if len_y < effective_length:
        #        new_y = np.zeros(effective_length, dtype=y.dtype)
        #        start = np.random.randint(effective_length - len_y)
        #        new_y[start:start + len_y] = y
        #        y = new_y.astype(np.float32)
        #    elif len_y > effective_length:
        #        start = np.random.randint(len_y - effective_length)
        #        y = y[start:start + effective_length].astype(np.float32)
        #    else:
        #        y = y.astype(np.float32)
        #labels = np.zeros(len(self.category_codes), dtype="f")
        # Does this set the label for the current code to 1 and all others to zero?
        #labels[self.category_codes[ebird_code]] = 1

        return {"spectros": spectros,
                "primary_target": row.primary_label,
                "all_targets": row.all_targets}

## Data Preprocess
# This is part of the PANNsCNN14Att model in the 'Intro to Sound Event Detection'
# module. 
# I want to first see what the output looks like for data using this preprocessing,
# then I can hook it up to the model and do all of that.
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


    # Get a spectrogram using the torch librosa functions.
    # If I want to do augmentations, create another Class/Module to do that.
    def spectro_from_data(self, input, mixup_lambda=None):
        spectrogram = self.spectrogram_maker(input)
        logmel_spectro = self.logmel_extractor(spectrogram)
        frequencies = dft_frequencies(self.window_size, self.sample_rate, self.bins)
        # self.spectro_plot(logmel_spectro, frequencies)
        return logmel_spectro


    def spectro_plot(self, logmel_spectro, frequencies):
        # 4d, but just need the innermost arrays transposed.
        ten_seconds = math.ceil((self.sample_rate / self.hop_size) * 10)
        reformat = logmel_spectro.detach().numpy()[0, 0, 0:ten_seconds].transpose(1,0)
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
    # I downsampled to 16000. 
    # Window size (1024) does not need to be that big; The lowest
    # frequency being 32+ Hz is not an issue. 
    preprocessor = SpectrogramCreator(sample_rate, 1024, 512, 64, 50, 8000)
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

# What do I actually want here?
# I don't see an inherint problem with using a dataframe, so stick with that 
# for now. Make sure to get all the labels.
def get_file_paths():
    # Get all the file paths, and assign each bird code name a number
    file_paths = []
    bird_labels = {}
    bird_number = 0;
    data_csv = pd.read_csv(resample_audio_csv_file)
    data_csv['all_labels'] = np.empty((len(data_csv), 0)).tolist()
    # Get the codes for the primary species in the recording, and add all the species
    # names to the label dictionary for use later.
    for idx, row in data_csv.iterrows():
        ebird_code = row.ebird_code
        species_name = row.species
        if ebird_code not in bird_labels:
           bird_labels[ebird_code] = bird_number
           bird_labels[species_name] = bird_number
           bird_number += 1
        resampled_filename = row.resampled_filename
        data_path = resample_audio_dir / ebird_code / resampled_filename
        file_paths.append([bird_labels[row.ebird_code], row.ebird_code, resampled_filename, data_path])

    # Get all the secondary labels, since the species names will now be in 
    # the bird_labels dict.
    for idx, row in data_csv.iterrows():
        row.all_labels.append(file_paths[idx][0])
        for bird in literal_eval(row.secondary_labels):
            species = re.sub(r'[^_]*_', '', bird)
            if species in bird_labels:
                row.all_labels.append(bird_labels[species])
        file_paths[idx].append(row.all_labels)

    return pd.DataFrame(file_paths, columns=["primary_label", "label_text", "resampled_filename", "resampled_full_path", "all_labels"]), bird_labels


# Add a column that labels each row with the iteration to use it for validation.
# With 5 splits, about 20% of the data is for validation each time, so
# THe labels in that column are unique (0, 1, 2, 3, 4).
def kfold_sampling(songfiles_metadata, target_varname, splits=5):
    songfiles_metadata['validation_fold'] = -1
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    # Inputs to skf.split() are x and y data. y can be considered labels.
    testsplit = skf.split(songfiles_metadata, songfiles_metadata[target_varname])
    for fold_id, (train_indices, val_indices) in enumerate(testsplit):
        songfiles_metadata.loc[val_indices, 'validation_fold'] = fold_id
    return songfiles_metadata


# How to do this:
# 1. Pass in all the data. Use the kfold_sampling to get a column that will
#    indicate what the splits are.
# 2. I won't be able to use Catalyst because its data loading/sampling specification
#    has no internal handling for k-folds.
# 3. Could try something like skorch, since it seems to have this functionality. 
#    I would then have to figure that out from scratch thought.
# 4. For my first go, maybe just do a raw pytorch loop and then try one of the
#    frameworks afterward since I will know what I am doing specifically.
def train_loop(all_data, paths, bird_codes):
   epoch_size = 5
   # paths, bird_codes = get_file_paths()
   # sample_data = kfold_sampling(paths, 'ebird_code', splits)
   spectrogram_maker = SpectrogramCreator(sample_rate, 1024, 256, 64, 50, 8000)
   for split in range(max(all_data['validation_fold'])):
       valid_data = all_data.loc[all_data['validation_fold'] == split]
       train_data = all_data.loc[all_data['validation_fold'] != split]
       valid_dataset = AudioDataset(valid_data, bird_codes, epoch_size, spectrogram_maker)
       train_dataset = AudioDataset(train_data, bird_codes, epoch_size, spectrogram_maker)
       # Test whether this works
       train_dataset.__getitem__(0)
       train_dataset.__getitem__(1)
       train_dataset.__getitem__(2)
       train_dataset.__getitem__(3)
       valid_dataset.__getitem__(0)
       valid_dataset.__getitem__(1)



if __name__ == "__main__":
    # explore_data()
    songfiles_metadata, bird_codes = get_file_paths()
    import pdb; pdb.set_trace()
    songfiles_metadata = kfold_sampling(songfiles_metadata, 'primary_label', 5)
    train_loop(songfiles_metadata, songfiles_metadata, bird_codes)
    print('Finished with data grabbing, sampling')
