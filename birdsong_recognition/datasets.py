import pandas as pd
import numpy as np
import soundfile as sf
import torch
import random
import math
from torch.utils.data import Dataset
from typing import List, Dict



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
            epoch_size,
            spectro_window_size,
            samples=5,
            is_training=True,
            waveform_transforms=None):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.category_codes = category_codes
        self.waveform_transforms = waveform_transforms
        self.is_training = is_training
        # This is the length of epoch?
        # Is there any way to get anything except the first epoch?
        self.epoch_size = epoch_size
        self.spectro_window_size = spectro_window_size
        self.samples = samples


    # This must be implemented
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
    def get_training_sample(self, data, sr, row, samples=5):
        # total observations given the config
        # total_obs = sr * epoch_size *  samples
        complete_epochs = int(len(data) // self.epoch_obs)
        time_series = []
        if complete_epochs >= samples:
            sampled_epochs = random.sample(list(range(complete_epochs)), samples)
            sample_idxs = [(epoch * self.epoch_obs, epoch * self.epoch_obs + self.epoch_obs) for epoch in sampled_epochs]
            for (start, end) in sample_idxs:
                time_series.extend(data[start:end])
            # print('Had enough epochs, length is: ', len(time_series))
        else:
            shortfall = self.total_obs - len(data)
            # print('Complete Epoch shortfall: ', samples - complete_epochs)
            time_series.extend(data)
            time_series.extend(np.zeros(shortfall))
            # print('Did no have enough epochs, length is: ', len(time_series))
        # Unsqueeze (nest the time series) because torch librosa 
        # spectrogram maker uses conv1d, 
        epoch_tensor = torch.FloatTensor(time_series)
        series_unsqueeze = epoch_tensor.unsqueeze(0)
        # Return just a single spectrogram for the whole sequence right now.
        return series_unsqueeze
        #spectro = self.spectrogram_maker.spectro_from_data(epoch_unsqueeze)
        #print('Dims of the spectro: ', spectro.size(), '\n')
        #return spectro


    # When predicting, do I just get the whole set of data?
    # Or do I need to use the same length as the train set?
    # I would say same length, unless SED is flexible enough to use any length.
    # That seems like a logical way to do it, but I will have to determine
    # how the model fits in.
    # 1. It seems like an Event Detection model should be able to predict
    #    on each epoch (5s) and then summarize to the whole length by 
    #    averaging those, no matter how many there are
    # 2. Maybe having a length that evenly divides into the epochs will
    #    be important however, so perhaps I should do that here.
    def get_prediction_data(self, data, sr, epoch_limit=100):
        complete_epochs = int(len(data) // self.epoch_obs)
        # complete_epochs = int(len(data) / sr // epoch_size)
        analyze_epochs = epoch_limit if complete_epochs > epoch_limit else complete_epochs
        prediction_epochs = list(range(analyze_epochs))
        spectro_idxs = [(epoch * self.epoch_obs, (epoch + 1) * self.epoch_obs) for epoch in prediction_epochs]
        spectros = []
        for (start, end) in sample_idxs:
            epoch_data = data[start:end]
            epoch_tensor = torch.from_numpy(epoch_data).float()
            # See training data note about unsqueeze
            series_unsqueeze = epoch_tensor.unsqueeze(0)
            # spectros.append(self.spectrogram_maker.spectro_from_data(epoch_unsqueeze))
        return series_unsqueeze


    # Get the code, filename. Load the data, sample a set of epochs from the 
    def __getitem__(self, idx: int):
        # import pdb; pdb.set_trace()
        # currently using a Subset here, so need to access the dataset
        row = self.file_list.dataset.iloc[idx, :]
        y, sr = sf.read(row.resampled_full_path)
        # Calculate a # of epoch obs based on the greatest whole number of
        # observations that divide evenly by the spectrogram step
        bins_per_epoch = math.ceil(math.ceil((self.epoch_size * sr * self.samples) / self.spectro_window_size) / self.samples)
        self.total_obs = int(bins_per_epoch * self.samples * self.spectro_window_size)
        self.epoch_obs = int(bins_per_epoch * self.spectro_window_size)

        # Problems: I need to change this to deal with data files that are
        # too short, and to deal with the fact that I should not return
        # separate nested data.  
        # Pad with zeros as needed, and then make a single spectrogram
        # out of the result.
        # print('Audiodataset got: ', row.resampled_full_path)
        if(self.is_training):
            time_series = self.get_training_sample(y, sr, row, samples=5)
        else:
            time_series = self.get_prediction_data(y, sr, epoch_limit=100)

        # I did not think I needed one-hot vector labels for torch;
        # add these back in if it is not working.
        # all_labels = self.convert_labels_to_coded(len(spectros), row.all_labels)
        # primary_labels = self.convert_labels_to_coded(len(spectros), row.primary_label)

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

        return {"time_series": time_series,
                "primary_label": row.primary_label,
                "all_labels": row.all_labels}



