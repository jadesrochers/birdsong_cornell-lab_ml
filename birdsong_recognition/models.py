import torch
import torch.nn as nn
import torchvision.models as models
from torch_librosa import Spectrogram, LogmelFilterBank, SpecAugmentation

## Base Wrapper for a model available through torchvision, which has
# datasets, models setups, and image transforms for computer vision.
# 1. Starting with this model because it was the core of the competition
#    winners setup.
class DenseNet121(nn.Module):
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


## Then make the model that wraps it using attention and will set up the
# actual segment/epoch and clip wise predictions.
class PANNsDense121Att(nn.Module):
    def __init__(self, sample_rate: int, window_size: int, hop_size: int,
                 mel_bins: int, fmin: int, fmax: int, classes_num: int, apply_aug: bool, top_db=None):
        super().__init__()
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        self.interpolate_ratio = 32  # Downsampled ratio
        self.apply_aug = apply_aug

        # Spectrogram extractor
        # I thought the data loader would handle these things,
        # do I actually need them?
        # Yes - I think the difference is that when training the loader
        # is used due to train/test/validate and the many epochs
        # but for predictions just a single set of data is fed.
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.att_block = AttBlock(1024, classes_num, activation='sigmoid')


        self.densenet_features = models.densenet121(pretrained=True).features

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def cnn_feature_extractor(self, x):
        x = self.densenet_features(x)
        return x

    def preprocess(self, input_x, mixup_lambda=None):
        x = self.spectrogram_extractor(input_x)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.apply_aug:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training  and self.apply_aug and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        return x, frames_num

    def forward(self, input_data):
        input_x, mixup_lambda = input_data
        """
        Input: (batch_size, data_length)"""
        b, c, s = input_x.shape
        input_x = input_x.reshape(b*c, s)
        x, frames_num = self.preprocess(input_x, mixup_lambda=mixup_lambda)
        if mixup_lambda is not None:
            b = (b*c)//2
            c = 1
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

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)
        frame_shape =  framewise_output.shape
        clip_shape = clipwise_output.shape
        output_dict = {
            'framewise_output': framewise_output.reshape(b, c, frame_shape[1],frame_shape[2]),
            'clipwise_output': clipwise_output.reshape(b, c, clip_shape[1]),
        }

        return output_dict
