from pytorch_lightning import LightningModule, Trainer
import pandas as pd
import numpy as np
import torch
from torch.nn import Module
import torchvision.models as models
from spectrograms import SpectrogramCreator
from torch import nn


# Initialization for layers in general, initialize weights and bias.
# This allows using a particular initialization distribution, changing it.
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

# Initialization for batch normalization layers
def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

## Base Wrapper for a model available through torchvision, which has
# datasets, models setups, and image transforms for computer vision.
# 1. Starting with this model because it was the core of the competition
#    winners setup.
class DenseNet121(Module):
    def __init__(self, pretrained=True, num_classes=6):
        super().__init__()
        self.densenet = models.densenet121(pretrained=pretrained)
        self.densenet.classifier = torch.nn.Linear(1024, num_classes)
        self.num_classes = num_classes

    def forward(self, x):  # batch_size, 3, a, b
        bs, seq, c, h, w = x.shape
        x = x.reshape(bs*seq,c,h,w)
        x = self.densenet(x)
        x = x.reshape(bs, seq, self.num_classes)
        return x


## Using Conv1d layers to simulate an attention block.
# nn.Linear and nn.Conv1d with kernel, step == 1 are essentially the same,
# so when looking up implementations of Attn you can remember that aspect.
class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        # In Features should be number of spectrogram bands?
        # Out features should be number of bird classes.
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    # x from torch.sum() gives a total sum for each out_feature for the input.
    # cla and norm_att give outputs for each epoch of the input, with different
    # activation setups. Don't know the why of those particularly.
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


# This is a variant on a denseNet121
class BirdieModel121(LightningModule):
    def __init__(self, sample_rate: int, spectro_window_size: int, spectro_step_size: int, classes_num: int, apply_aug: bool, mel_bins: int=512, fmin: int=0, fmax: int=22050,):
        super().__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        self.interpolate_ratio = 32  # Downsampled ratio
        self.apply_aug = apply_aug
        self.sample_rate = sample_rate
        self.spectro_window_size = spectro_window_size
        self.spectro_step_size = spectro_step_size
        self.loss_bce = nn.BCELoss(reduction='none')
        self.classes_num = classes_num

        ## REASONS to do Transforms/Specto HERE -
        # Because I can do the logmel/spectro/augmentations on a GPU.
        # There are methods set up in the LightningModule to do operations
        # on data at very specific times, including after transfer to GPU/TPU,
        # and that would allow me to leverage the torchlibrosa in a way the
        # DataLoader cannot.

        # window_size is the spectrogram window size, not the analysis window.
        # step_size is the moving window jump for that spetrogram window.
        self.spectrogram_maker = SpectrogramCreator(self.sample_rate, self.spectro_window_size, self.spectro_step_size)

        # I want to use an Augmenter somewhere, the data loader may
        # be the more appropriate place.
        # If incorporating into the Model here, probably use this in
        # on_before_batch_transfer(), which is designed for data augmentation. 
        #self.spec_augmenter = SpecAugmentation(
        #    time_drop_width=64,
        #    time_stripes_num=2,
        #    freq_drop_width=8,
        #    freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(self.spectrogram_maker.mel_bins)
        self.fc1 = nn.Linear(self.spectro_window_size, self.spectro_window_size, bias=True)
        # THe Attn block needs to take the number
        self.att_block = AttBlock(1024, classes_num, activation='sigmoid')
        self.densenet_features = models.densenet121(pretrained=True).features
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    # This would probably be called in on_before_batch_transfer() which is 
    # designed for data_augmentations and modification before the data 
    # gets passed to the device (TPU/GPU).
    # If it will be better to do this after transferring to device
    # for parallel purposes, on_after_batch_transfer() is the method to impl.
    # Probably better after transfer; take advantage of the GPU.
    def cnn_feature_extractor(self, x):
        x = self.densenet_features(x)
        return x


    def get_binary_labels(self, indices):
        labels = []
        for i, index_set in enumerate(indices):
           labels.append(np.zeros(self.classes_num, dtype="f"))
           for index in index_set:
               labels[i][index] = 1
        return labels

    # Do processing on the GPU but before doing train part
    # I put the spectrogram generation here to leverage gpu/torchlibrosa
    def on_after_batch_transfer(self, batch_data, batch_idx):
        # If the whole batch is passed, may need to do this all in a 
        # loop or something
        print('Dims of the batch_data: ', len(batch_data))
        spectro = []
        for input in batch_data:
          spectro.append(self.spectrogram_maker.spectro_from_data(input['time_series']))
        primary_labels = [[data['primary_label']] for data in batch_data ]
        all_labels = [data['all_labels'] for data in batch_data ]
        import pdb; pdb.set_trace()
        primary_binary_labels = self.get_binary_labels(primary_labels)
        all_binary_labels = self.get_binary_labels(all_labels)
        print('Number of mel spectrograms: ', len(spectro), '\n')
        # Pass along the labels until I don't need them.
        return {'spectro': spectro,
                'primary_label': primary_binary_labels,
                'all_labels': all_binary_labels}


    # Forward. Should get passed just the input data by the 
    # train step, which will take the results from this and calc loss.
    def forward(self, input_spectros):
        """
        Input: (batch_size, data_length)"""
        b, c, s = input_spectros.shape
        x = input_spectros.reshape(b*c, s)

        # Output shape (batch size, channels, time, frequency)
        x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
        x = self.cnn_feature_extractor(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        # The attn block takes the output from the densenet121 with a few 
        # operations done to it but the dimensions unmodified.
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        frame_shape =  framewise_output.shape
        clip_shape = clipwise_output.shape
        # Framewise/segment is each section of data, clipwise is prediction 
        # for the whole audio clip.
        output_dict = {
            'framewise_output': framewise_output.reshape(b, c, frame_shape[1],frame_shape[2]),
            'clipwise_output': clipwise_output.reshape(b, c, clip_shape[1]),
        }

        return output_dict


    # The purpose of this is to return loss
    def training_step(self, batch, batch_idx):
        # You can use the epoch/subsegment data, the full recording data,
        # or some combination of both. You need to decide what make the
        # most sense regarding labels and segments.
        import pdb; pdb.set_trace()
        spectros, primary_labels, all_labels = batch
        y_hat = self(spectros)
        # Simple method right now is to just use primary labels
        loss = self.loss_bce(y_hat, primary_labels)
        return loss


    # just doing a direct copy of the training_step at the moment,
    # need to get this going to see some predictions.
    def validation_step(self, batch, batch_idx):
        spectros, primary_labels, all_labels = batch
        y_hat = self(spectros)
        # Simple method right now is to just use primary labels
        loss = self.loss_bce(y_hat, primary_labels)
        return loss


    def configure_optimizers(self):
        # I don't have any configuration params at the moment
        # return torch.optim.Adam(self.config_params, lr=0.01)
        return torch.optim.Adam(self.parameters(), lr=0.01)


