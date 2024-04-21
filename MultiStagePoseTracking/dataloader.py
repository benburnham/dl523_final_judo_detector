'''
Data Loader JUDO Identifier
Ben Burnham
April 21, 2024
Deep Learning EC523
'''
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video

class VideoDataset(Dataset):
    def __init__(self, root_dir, mode='train', include_mirror=True, include_audio=True):
        self.root_dir = root_dir
        self.mode = mode
        self.include_mirror = include_mirror
        self.include_audio = include_audio

        if include_mirror:
            self.sub_dirs = ['ALL', 'NO_MIRROR']
        else:
            self.sub_dirs = ['ALL']

        self.data = []
        for sub_dir in self.sub_dirs:
            sub_dir_path = os.path.join(self.root_dir, sub_dir, mode)
            techniques = os.listdir(sub_dir_path)
            for technique in techniques:
                technique_path = os.path.join(sub_dir_path, technique)
                videos = [os.path.join(technique_path, video) for video in os.listdir(technique_path)]
                self.data.extend(videos)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx]
        video, audio, info = read_video(video_path, include_audio=self.include_audio)
        # You may need to do further preprocessing here depending on your requirements
        if self.include_audio:
            return video, audio
        else:
            return video