import pandas as pd
import numpy as np
import soundfile as sf
import torch
import random
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

    # Important; this must be implemented
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
    def get_training_sample(self, data, sr, row, samples=5, epoch_size=5):
        # total observations given the config
        total_obs = sr * epoch_size *  samples
        complete_epochs = int(len(data) / sr // epoch_size)
        time_series = []
        if complete_epochs >= samples:
            sampled_epochs = random.sample(list(range(complete_epochs)), samples)
            sample_idxs = [(epoch * sr, (epoch + 1) * sr) for epoch in sampled_epochs]
            for (start, end) in sample_idxs:
                time_series.extend(data[start:end])
        else:
            shortfall = total_obs - len(data)
            print('Data row : ', row, ' has shortfall: ', shortfall)
            print('Complete Epoch shortfall: ', samples - complete_epochs)
            time_series.extend(data)
            time_series.extend(np.zeros(shortfall))
        # Unsqueeze (nest the time series) because the spectrogram maker 
        # uses a conv1d, which expects a set of series, even if the set == 1.
        epoch_tensor = torch.FloatTensor(time_series)
        epoch_unsqueeze = epoch_tensor.unsqueeze(0)
        # Return just a single spectrogram for the whole sequence right now.
        spectro = self.spectrogram_maker.spectro_from_data(epoch_unsqueeze)
        return spectro

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
        # Problems: I need to change this to deal with data files that are
        # too short, and to deal with the fact that I should not return
        # separate nested data.  
        # Pad with zeros as needed, and then make a single spectrogram
        # out of the result.
        if(self.is_training):
            spectros = self.get_training_sample(y, sr, row, samples=5, epoch_size=5)
        else:
            spectros = self.get_prediction_data(y, sr, epoch_limit=100)

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

        return {"spectros": spectros,
                "primary_label": row.primary_label,
                "all_labels": row.all_labels}



