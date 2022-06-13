from pytorch_lightning import LightningModule, Trainer
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
from spectrograms import SpectrogramCreator
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from models import DenseNet121


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


def interpolate(x: torch.Tensor, ratio: int, raw_length: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    # If length is slightly off, make it exactly the same as before.
    batch, time, classes = upsampled.shape
    pad = torch.zeros(batch, raw_length - time, classes).cuda(upsampled.get_device())
    if pad.shape[0] > 0:
        upsampled = torch.cat((upsampled, pad), 1)
    # upsampled = F.pad(upsampled, raw_length,'constant',0)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


## Base Wrapper for a model available through torchvision, which has
# datasets, models setups, and image transforms for computer vision.
# 1. Starting with this model because it was the core of the competition
#    winners setup.
# class DenseNet121(Module):
#    def __init__(self, pretrained=True, num_classes=6):
#        super().__init__()
#        self.densenet = models.densenet121(pretrained=pretrained)
#        self.densenet.classifier = torch.nn.Linear(1024, num_classes)
#        self.num_classes = num_classes
#
#    def forward(self, x):  # batch_size, 3, a, b
#        bs, seq, c, h, w = x.shape
#        x = x.reshape(bs*seq,c,h,w)
#        x = self.densenet(x)
#        x = x.reshape(bs, seq, self.num_classes)
#        return x

# Do this earlier on, since you need these labels for any BCE
def convert_labels_to_binary(classes, numeric_labels, device):
    binary_labels = torch.zeros(len(numeric_labels), classes).cuda(device)
    for batch_idx, labels in enumerate(numeric_labels):
        for label in labels:
            binary_labels[batch_idx][label] = 1
    return binary_labels


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
        cla = torch.softmax(self.cla(x), dim=-1)
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    # When this is used, loss was not working because it was not 0-1
    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)




#for clip_out in y_hat['clipwise_output']:
#    for num in clip_out:
#        if num <=0 or num >=1:
#            print('num was out of range: ', num)


# This is a variant on a denseNet121
class BirdieModel121(LightningModule):
    def __init__(self, sample_rate: int, spectro_window_size: int, spectro_step_size: int, num_classes: int, apply_aug: bool, mel_bins: int=64, fmin: int=0, fmax: int=22050,):
        super().__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        # DenseNet121 does 2 stride down sampling 5 times, so /32
        # in terms of downsampling ratio 
        self.downsample_ratio = 32
        self.apply_aug = apply_aug
        self.sample_rate = sample_rate
        self.spectro_window_size = spectro_window_size
        self.spectro_step_size = spectro_step_size
        self.loss_bce = nn.BCELoss(reduction='mean')
        self.num_classes = num_classes

        ## REASONS to do Transforms/Specto HERE -
        # Because I can do the logmel/spectro/augmentations on a GPU,
        # unlike the DataLoader, which is not ready-made to use GPU.

        # The dataset will automatically pick an epoch/data size
        # to match the spectro step, which should be a power of 
        # 2 to make sure the fft is in use.
        # TODO: I could try and also set the step to the total # of steps
        # will divide evenly by the model down/up sample, though
        # this will heavily limit the steps I can choose.


        # step_size is the moving window jump for spectrogram calculation.
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
        # The Attn block needs to take the number
        self.att_block = AttBlock(1024, num_classes)
        # get rid of this, use the wrapped version from models module.
        # self.densenet_features = DenseNet121(pretrained=True, num_classes=num_classes).densenet.features
        self.densenet_features = models.densenet121(pretrained=True).features
        # DenseNet121(pretrained=True, num_classes=num_classes).densenet.features
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)


    # Called as part of forward.
    def cnn_feature_extractor(self, x):
        # Do I need to put this in eval, or do something else?
        x = self.densenet_features(x)
        return x


    def get_binary_labels(self, indices):
        labels = []
        for i, index_set in enumerate(indices):
           labels.append(np.zeros(self.num_classes, dtype="f"))
           for index in index_set:
               labels[i][index] = 1
        return labels


    # Do processing on the GPU but before doing train part
    # I put the spectrogram generation here to leverage gpu/torchlibrosa
    def on_after_batch_transfer(self, batch_data, batch_idx):
        print('Input data batch size: ', len(batch_data))
        spectros = torch.tensor([]).cuda(self.device)
        for input in batch_data:
            # print('Input data series shape: ', input['time_series'].shape)
            new_spectro = self.spectrogram_maker.spectro_from_data(input['time_series'])
            # print('Spectro shape: ', new_spectro.shape)
            spectros = torch.cat((spectros, new_spectro))
        spectros = torch.squeeze(spectros)
        spectros = torch.unsqueeze(spectros, dim=1)
        primary_labels = [[data['primary_label']] for data in batch_data ]
        all_labels = [data['all_labels'] for data in batch_data ]
        # Tensors of binary labels for all/primary, and secondary if you find a use for that.
        primary_binary_labels = convert_labels_to_binary(self.num_classes, primary_labels, self.device)
        all_binary_labels = convert_labels_to_binary(self.num_classes, all_labels, self.device)
        print('Mel spectrograms dims: ', spectros.size(), '\n')
        # Pass along the labels until I don't need them.
        return {'spectros': spectros,
                'primary_labels': primary_binary_labels,
                'all_labels': all_binary_labels}


    # The purpose of this is to return loss, typically by calling 
    # self() which invokes the forward() method.  
    def training_step(self, batch, batch_idx):
        # You can use the epoch/subsegment data, the full recording data,
        # or some combination of both. You need to decide what make the
        # most sense regarding labels and segments.
        spectros = batch['spectros']
        primary_labels  = batch['primary_labels']
        all_labels = batch['all_labels']

        y_hat = self(spectros)
        # Simple method right now is to just use primary labels
        loss = self.loss_bce(y_hat['clipwise_output'], primary_labels)
        # import pdb; pdb.set_trace()
        return loss


    # just doing a direct copy of the training_step at the moment,
    # need to get this going to see some predictions.
    def validation_step(self, batch, batch_idx):
        spectros = batch['spectros']
        primary_labels  = batch['primary_labels']
        all_labels = batch['all_labels']

        y_hat = self(spectros)
        # Simple method right now is to just use primary labels
        loss = self.loss_bce(y_hat['clipwise_output'], primary_labels)
        return loss


    # Forward. Should get passed just the input data by the 
    # train step, which will take the results from this and calc loss.
    def forward(self, input_spectros):
        """
        Input: (batch_size, data_length)
        """
        # x = input_spectros.reshape(b*c, s)
        # NOTE: It seems that WxH is fine too; the important thing is that
        # you keep track of which is which.
        # Get the spectrograms in terms of HxW
        # x = torch.transpose(input_spectros, 2, 3)
        batchsz, emptysz, melW, melH = input_spectros.shape

        # Output shape (batch size, channels, time, frequency)
        # x = x.expand(x.shape[0], 3, x.shape[1], x.shape[2])
        x = input_spectros.expand(batchsz, 3, melW, melH)
        x = self.cnn_feature_extractor(x)

        # Aggregate in frequency axis - isn't the densenet already 
        # doing this?
        # This seems to be only if the densenet does not fully
        # collapse the frequency dimension, which if I do mel_spec
        # bins=40, it will already be collapsed.
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
        # If the dimensions are correct, the clipwise_output should collapse
        # the 3rd dimension, time.
        # The segmentwise_output should preserve it, but with reduced 
        # resolution according to the parameters of the densenet121.  
        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output - The interpolation should take the reduced
        # time dimension and re-expand it to the original size.
        # Use the downsampling and raw size to re-expand.
        framewise_output = interpolate(segmentwise_output,
                                       self.downsample_ratio, melW)
        # framewise_output = pad_framewise_output(framewise_output, frames_num)

        # The outputs should be standardized in shape, with one having
        # an output for each spectrogram frame time step, and the clip
        # summarizing over the whole clip.
        frame_shape =  framewise_output.shape
        clip_shape = clipwise_output.shape
        # Framewise/segment is each section of data, clipwise is prediction 
        # for the whole audio clip.

        # TODO: might need to reshape these, but for now just going to pass
        # them along until I determine how
        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output
        }

        return output_dict


    def configure_optimizers(self):
        # I don't have any configuration params at the moment
        # return torch.optim.Adam(self.config_params, lr=0.01)
        return torch.optim.Adam(self.parameters(), lr=0.01)


