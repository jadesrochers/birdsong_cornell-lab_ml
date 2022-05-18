
from pathlib import Path
from lightning_data_modules import BirdieDataModule
from lightning_model_modules import BirdieModel121
from pytorch_lightning import Trainer

root_dir = Path.cwd()
resample_audio_csv_file =  root_dir / 'train_resample.csv'
resample_audio_dir =  root_dir / 'train_audio_resampled'

sample_rate = 16000
spectro_window_size = 1024
spectro_step_size = 256
# size in seconds
epoch_size = 5
batch_size = 32
classes = 264
mel_bins = 40
# import pdb; pdb.set_trace()
birdie_model = BirdieModel121(sample_rate, spectro_window_size, spectro_step_size,classes, False, mel_bins)
birdie_datamodule = BirdieDataModule(resample_audio_csv_file, resample_audio_dir, sample_rate, spectro_window_size, epoch_size, batch_size)
birdie_datamodule.prepare_data()
birdie_datamodule.setup()
validation_loaders = birdie_datamodule.val_dataloader()
train_loaders = birdie_datamodule.train_dataloader()
# This is how to get data out of a loader.
#for batch_idx, spectro in enumerate(validation_loaders[0]):
    # import pdb; pdb.set_trace()
#    break;
    # print('At index : ', batch_idx, ' with spectro: ', spectro)

#for batch_idx, spectro in enumerate(validation_loaders[3]):
    # import pdb; pdb.set_trace()
#    break;
    # print('At index : ', batch_idx, ' with spectro: ', spectro)

#for batch_idx, spectro in enumerate(train_loaders[0]):
    # import pdb; pdb.set_trace()
    # if batch_idx > 0:
#    break;
    # print('At index : ', batch_idx, ' with spectro: ', spectro)

print('Done with testing the data module')

trainer = Trainer(gpus=1, max_epochs=2)
trainer.fit(birdie_model, birdie_datamodule)



### The Overall Plan:

# 1. Sample sound data from each file. We need the same amount from each
#    file, so I may need to pre-emptively exclude files with too little
#    data while also zero padding those with a good amount but 
#    needing a filling out.
# 1A. Two step analysis? - I have been thinking that the training may
#     have a problem with training on blank or irrelevant stuff, and 
#     if I only have it train on relevant signals it will do much better.
#     Can incorporate this when you actually get it working.
# 1B. I could also add an 'other' or set of categories that allows
#     the model to identify sounds to specifically ignore, without
#     giving them precise labels.

# 2. The sample segments should probably be sequential because
#    The data is best combined into one time series and then translated
#    into spectrogram.

# 3. Make the spectrogram in the data genertor or model preprocess,
#    but make sure that each file has one continuous spectrogram.

# 4. The Densenet input will be batch_size x 3 x spectro_steps x 
#    spectro_bands.
#    The '3' part is confusing; I think it needs that even if the input
#    is single channel. The rest makes sense; one spectro for each
#    file in the batch, with the step/window/sample determining the rest.

# 5. The densenet feature output can be used; for a densenet121 this
#    will be spectro_steps/16 with 1024 channels. See densenet diagram
#    To understand that, but it ends up with a lot of channels due to
#    always compounding the input to the next layer within a dense block.

# 6. Then use that output with a few extra layers to get the 
#    classification values for each segment and the whole input.

# 7. Clarify: you will need to make sure you can track time despite all
#    the condensing that the densenet121 will do, and I am not sure
#    of that yet, will need to look at the example more for that.
