## Put Lighting Data and Model modules here.
# You will need to pull in the base models, dataset/loaders from elsewhere,
# but lightning will serve to organize and run everything.

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import re
from ast import literal_eval
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from spectrograms import SpectrogramCreator
from datasets import AudioDataset

# Lightning Data Module.
# I like the organization, and it has helped make it clearer how I get this 
# data organized, setup, and moving along, so I am going to go with this for now.
class BirdieDataModule(LightningDataModule):


    def __init__(self, csv_path, data_path, sample_rate, spectro_window_size, epoch_size=5, batch_size=32):
        self.resampled_audio_csv_filepath = csv_path
        self.resampled_audio_dir = data_path
        self.sample_rate = sample_rate
        self.epoch_size = epoch_size
        self.spectro_window_size = spectro_window_size
        self.batch_size = batch_size
        self.songfiles_metadata = pd.DataFrame()
        self.bird_codes = {}
        self.valid_dataloaders = []
        self.train_dataloaders = []
        self.all_dataloader = None


    # Data prep to be done, but not in a multi-process way
    def prepare_data(self):
        self.songfiles_metadata, self.bird_codes = self.get_file_paths()
        # self.songfiles_metadata = self.kfold_sampling(songfiles_metadata, 'primary_label', 5)


    # Setup is performed before any of the dataloader methods are called.
    # will be distributed across gpus if possible.
    # If I want to do a train/test split, this would be a good place.
    # not doing that right now.
    # def setup(self, stage=None):
    #     # For fit, get train/validate data.
    #     if stage == "fit" or stage is None:
    #         self.make_datasets(self.songfiles_metadata, self.bird_codes)

    
    # Make the folds, but don't do anything with them yet.
    # Use stratified kfold since grous are small and I don't want to 
    # skip one on various trains.
    def setup_folds(self, target_varname, splits=5):
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
        # Inputs to skf.split() are the whole dataframe and the 
        # labels to stratify by.
        splitter = skf.split(self.songfiles_metadata, self.songfiles_metadata[target_varname])
        self.splits = [split for split in splitter]
 

    def setup_fold_index(self, fold_index):
        train_indices, val_indices = self.splits(fold_index)
        self.train_fold = Subset(self.songfiles_metadata, train_indices)
        self.val_fold = Subset(self.songfiles_metadata, val_indices)


    # def kfold_sampling(self, songfiles_metadata, target_varname, splits=5):
    #     songfiles_metadata['validation_fold'] = -1
    #     skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    #     # Inputs to skf.split() are x and y data. y can be considered labels.
    #     testsplit = skf.split(songfiles_metadata, songfiles_metadata[target_varname])
    #     # Add labels to the metadata DF indicating which fold data are part of
    #     # In 0 to n-1 terms.
    #     for fold_id, (train_indices, val_indices) in enumerate(testsplit):
    #         songfiles_metadata.loc[val_indices, 'validation_fold'] = fold_id
    #     return songfiles_metadata



    # def make_datasets(self, all_data, bird_codes):
    #     self.all_dataloader = DataLoader(AudioDataset(all_data, bird_codes, self.epoch_size, self.spectro_window_size), batch_size=self.batch_size, shuffle=True)
    #     # Make as many Datasets as you have k-folds for train and valid data. 
    #     for split in range(max(all_data['validation_fold'])):
    #         valid_data = all_data.loc[all_data['validation_fold'] == split]
    #         train_data = all_data.loc[all_data['validation_fold'] != split]
    #         valid_dataset = AudioDataset(valid_data, bird_codes, self.epoch_size, self.spectro_window_size)
    #         train_dataset = AudioDataset(train_data, bird_codes, self.epoch_size, self.spectro_window_size)
    #         # The collate_fn seems to be needed due to me returning
    #         # a map of data as opposed to a simple array/tensor.
    #         self.valid_dataloaders.append(DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: x))
    #         self.train_dataloaders.append(DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: x))



    # Don't know If arbitrary methods are fine, but going to package
    # these in here and see.
    def get_file_paths(self):
        # Get all the file paths, and assign each bird code name a number
        file_paths = []
        bird_labels = {}
        bird_number = 0;
        data_csv = pd.read_csv(self.resampled_audio_csv_filepath)
        data_csv['all_labels'] = np.empty((len(data_csv), 0)).tolist()
        # Get the codes for the primary species in the recording, 
        # and add all the species names to the label dictionary 
        # for use later.
        for idx, row in data_csv.iterrows():
            ebird_code = row.ebird_code
            species_name = row.species
            if ebird_code not in bird_labels:
               bird_labels[ebird_code] = bird_number
               bird_labels[species_name] = bird_number
               bird_number += 1
            resampled_filename = row.resampled_filename
            data_path = self.resampled_audio_dir / ebird_code / resampled_filename
            file_paths.append([bird_labels[row.ebird_code], row.ebird_code, resampled_filename, data_path])

        # Get all the secondary labels, since the species names 
        # will now be in # the bird_labels dict.
        for idx, row in data_csv.iterrows():
            row.all_labels.append(file_paths[idx][0])
            for bird in literal_eval(row.secondary_labels):
                species = re.sub(r'[^_]*_', '', bird)
                if species in bird_labels:
                    row.all_labels.append(bird_labels[species])
            file_paths[idx].append(row.all_labels)

        return pd.DataFrame(file_paths, columns=["primary_label", "label_text", "resampled_filename", "resampled_full_path", "all_labels"]), bird_labels


    # Handles loading all the training data. If I want to be consistent, will
    # Have to make separate train loaders for each validation split.
    def train_dataloader(self):
        return AudioDataset(self.train_fold, self.bird_codes, self.epoch_size, self.spectro_window_size)

    # Validation data. This can take an array of dataloaders, so for k-folds
    # I think I will try that.
    def val_dataloader(self):
        return AudioDataset(self.val_fold, self.bird_codes, self.epoch_size, self.spectro_window_size)
        # return self.valid_dataloaders

    # Test data is a set put aside to test the model once it has completed 
    # training, not during training like validation. 
    # Not going to do this right now.
    # def test_dataloader(self):

    # I will likely just predict on the whole set trained on, 
    # I am just trying to get it working right now
    def predict_dataloader(self):
        return AudioDataset(self.songfiles_metadata, self.ebird_codes, self.epoch_size, self.spectro_window_size)


