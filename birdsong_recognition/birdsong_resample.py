## Resampling: 
# Some of the projects indicated this was a good idea. I might do it just to get a feel for 
# what the data is like and how a change in sampling might affect things.

import argparse
import pandas as pd
import librosa
import math
import soundfile as sf
from functools import partial
from pathlib import Path
from multiprocessing import Pool
import pdb

def split_dataframe(resample_limit, df, num_chunks=6):
   df = df.iloc[0:resample_limit, :]
   split_dfs = []
   df_size = math.ceil(len(df) / num_chunks)
   end = df_size
   start = 0
   for i in range(num_chunks):
      if end > len(df): end = len(df)
      split_dfs.append(df.iloc[start:end, :])
      start = end
      end = end + df_size
   return split_dfs

# This will take one or more channels of input at various sampling rates and 
# turn them into a single channel at the specified rate
def resample(resample_to: int, df: pd.DataFrame):
    raw_audio_dir = Path("train_audio")
    resample_dir = Path("train_audio_resampled")
    resample_dir.mkdir(exist_ok=True, parents=True)
    for i, row in df.iterrows():
        ebird_code = row.ebird_code
        filename = row.filename
        ebird_resamp_dir = resample_dir / ebird_code
        if not ebird_resamp_dir.exists():
            ebird_resamp_dir.mkdir(exist_ok=True, parents=True)
        try:
            y, _ = librosa.load(
                raw_audio_dir / ebird_code / filename,
                sr=resample_to, mono=True, res_type="kaiser_fast")
            filename = filename.replace(".mp3", ".wav")
            print('Current bird, file: ', ebird_code, filename)
            sf.write(ebird_resamp_dir / filename, y, samplerate=resample_to)
        except Exception:
            with open("skipped.txt", "a") as f:
                file_path = str(raw_audio_dir / ebird_code / filename)
                f.write(file_path + "\n")

# command line resampler. Takes arguments for the sample rate and how many operations to run in parallell.
if __name__ == "__main__":
    # get the args. Sampling rate and how many processes to spin off.
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--sr", default=16000, type=int)
    parser.add_argument("--cores", default=6, type=int)
    args = parser.parse_args()

    # Break the train file/df up into pieces to do some multi-processing
    resample_to = args.sr
    multi_cores = args.cores
    resample_limit = args.limit
    train_csv = pd.read_csv("train.csv")
    split_dfs = split_dataframe(resample_limit, train_csv, multi_cores)

    # Run the resample method using the number of processes, giving them each one sub df to work with.
    resample_configured = partial(resample, resample_to)
    with Pool(processes=multi_cores) as pool:
        pool.map(resample_configured, split_dfs, 1)

    # Save a new file with the added data
    train_csv["resampled_sampling_rate"] = resample_to
    train_csv["resampled_filename"] = train_csv["filename"].map(
        lambda x: x.replace(".mp3", ".wav"))
    train_csv["resampled_channels"] = "1 (mono)"
    train_csv.to_csv("train_resample.csv", index=False)


## Example resampling code can be found here:
# https://github.com/koukyo1994/kaggle-birdcall-resnet-baseline-training/blob/f89d49b6b34c9d0b9a6f516d5154b33caa4ac911/input/birdsong-recognition/prepare.py
