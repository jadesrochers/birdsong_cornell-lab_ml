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
from spectrograms import SpectrogramCreator
from datasets import AudioDataset


root_dir = Path.cwd()
raw_audio_dir = root_dir / 'train_audio'
resample_audio_dir =  root_dir / 'train_audio_resampled'
resample_audio_csv_file =  root_dir / 'train_resample.csv'
train_csv_file = root_dir / 'train.csv'
train_csv_df = pd.read_csv(train_csv_file)


## These might get pulled into model data loaders, or preprocess methods.
# I want to first make sure they work right however, so start with them broken out.

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
def make_datasets(all_data, bird_codes):
   epoch_size = 5
   # paths, bird_codes = get_file_paths()
   # sample_data = kfold_sampling(paths, 'ebird_code', splits)
   spectrogram_maker = SpectrogramCreator(sample_rate, 1024, 256, 64, 50, 8000)
   valid_datasets = []
   train_datasets = []
   for split in range(max(all_data['validation_fold'])):
       valid_data = all_data.loc[all_data['validation_fold'] == split]
       train_data = all_data.loc[all_data['validation_fold'] != split]
       valid_datasets.add(AudioDataset(valid_data, bird_codes, epoch_size, spectrogram_maker))
       train_datasets.add(AudioDataset(train_data, bird_codes, epoch_size, spectrogram_maker))
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
    make_datasets(songfiles_metadata, songfiles_metadata, bird_codes)
    print('Finished with data grabbing, sampling')
