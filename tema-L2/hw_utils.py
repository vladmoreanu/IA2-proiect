import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import librosa
from IPython.display import Audio, display
import matplotlib.pyplot as plt

import numpy as np

import requests, zipfile, io, logging
from tqdm import tqdm
import os

zip_file_url = "https://zenodo.org/records/3871592/files/MiniLibriMix.zip?download=1"

def download_with_progress(url, destination):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 * 10   # 10MB

    with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))


def prep_dataset(path):
    zipfile_path = os.path.join(path, "MiniLibriMix.zip")
    if not os.path.exists(path):
        os.makedirs(path)
        download_with_progress(zip_file_url, zipfile_path)
        with zipfile.ZipFile(zipfile_path, 'r') as z:
            z.extractall(path)
        os.remove(zipfile_path)
        return


class LibriMixDataset(Dataset):
    def __init__(self, root, subset="train", segment_length=3, typ="clean", sample_rate=8000):
        """
        :param root: string, path to the folder containing MiniLibriMix directory
        :param subset: string, either "train" or "val"
        :param segment_length: float, the length of each sample in seconds
        :param typ: string, either "clean" for non-noisy mixtures, or "both" which includes the background noise
        """

        self.root = root
        self.subset = subset
        self.typ = typ
        self.sample_rate = sample_rate

        self.seg_len = segment_length * self.sample_rate
        
        assert self.subset in ["train", "val"]
        # assert self.typ in ["clean", "both", ]

        self.mixtures_path = os.path.join(self.root, 
                                          "MiniLibriMix",
                                          self.subset,
                                          f"mix_{self.typ}",)

        self.metadata_path = os.path.join(self.root, "MiniLibriMix/metadata", f"mixture_{self.subset}_mix_{self.typ}.csv")
        self.metadata = pd.read_csv(self.metadata_path)

        self.segments_per_file = self._segs_per_file()

        # We'll create a list of mixture paths, each one repeated by the number of segments it contains (and the limits for each segment)
        # This will help when indexing the dataset -- we'll know which file to read and from which timestamps to segment
        self.mixtures = []
        for m, k in self.segments_per_file.items():
            for i in range(k):
                self.mixtures.append(
                    (m, i * self.seg_len, (i + 1) * self.seg_len)  # (mixture_path, start_index, end_index)
                )

    def _segs_per_file(self):
        """
        Returns a dictionary with keys : <mixture file path> and values : <#segments per file>.
        This will help in assessing the dataset length (internally used by DataLoader to batch samples) and in constructing the
        mixtures list from which we'll sample given an index (key).
        """
        length_dict = {}
        for index, row in self.metadata.iterrows():
            mixture_path = row['mixture_path']
            length = row['length']
            if length < self.seg_len:
                length_dict[mixture_path] = 1   # we'll pad this with 0s
            else:
                length_dict[mixture_path] = int(length // self.seg_len)
    
        return length_dict

    def __len__(self):
        return sum(list(self.segments_per_file.values()))

    def __getitem__(self, key):
        """
        Retrieves the mixture and sources segments.
        """
        mix_path, sources_path, timestamps = self._get_metadata(key)

        start, end = timestamps
        
        mix, sr = librosa.load(mix_path, sr=self.sample_rate)
        mix = mix[start: end]
        sources = []
        for s_path in sources_path:
            source, sr = librosa.load(s_path, sr=self.sample_rate)
            source = source[start: end]
            sources.append(source)
        
        return mix, torch.FloatTensor(np.array(sources))

    def listen_samples(self, key):
        """
        Retrieves signal paths and plots waveform/audio widget.
        """
        mix_path, sources_path, timestamps = self._get_metadata(key)
        
        start, end = timestamps
        
        audio_files = [mix_path] + sources_path
        audio_titles = ['Mix', 'Source 1', 'Source 2']
        
        audio_widget_groups = []
        for audio_file, title in zip(audio_files, audio_titles):
            
            # Load the audio file and extract the signal and sampling rate
            signal, sr = librosa.load(audio_file, sr=self.sample_rate)

            signal = signal[start: end]
            # Plot the waveform
            plt.figure(figsize=(10, 1))
            librosa.display.waveshow(signal, sr=self.sample_rate)
            plt.title(title)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.ylim([-1, 1])
            plt.show()
            
            # Display the audio player
            display(Audio(data=signal, autoplay=False, rate=self.sample_rate))
    
    def _get_metadata(self, key):
        """
        Retrieves the paths for one element of the dataset:
            - mixture
            - source 1
            - source 2

        And returns these paths along with the start-end timestamps
        """
        
        mixture_path, start, end = self.mixtures[key]

        row = self.metadata[self.metadata["mixture_path"] == mixture_path]
        s1_path = row["source_1_path"].values[0]
        s2_path = row["source_2_path"].values[0]

        mixture_path = os.path.join(self.root, mixture_path)
        s1_path = os.path.join(self.root, s1_path)
        s2_path = os.path.join(self.root, s2_path)
        
        return mixture_path, [s1_path, s2_path], (start, end)