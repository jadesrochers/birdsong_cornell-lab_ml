
from pathlib import Path
from lightning_modules import BirdieDataModule

import pdb; pdb.set_trace()
root_dir = Path.cwd()
resample_audio_csv_file =  root_dir / 'train_resample.csv'
resample_audio_dir =  root_dir / 'train_audio_resampled'

sample_rate = 16000
# size in seconds
epoch_size = 5
test_datamod = BirdieDataModule(resample_audio_csv_file, resample_audio_dir, sample_rate, epoch_size)
test_datamod.prepare_data()
