import os
import gc
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_CONTEXT = 60

def read_file(path):
    return np.load(path)

def padding(context):
    return np.zeros((MAX_CONTEXT, 28))

def pad(arr, context):
    return np.concatenate([padding(context), arr, padding(context)])

def locate_line(index_mapping, ind):
    for i in range(len(index_mapping)-1):
        if index_mapping[i] <= ind < index_mapping[i+1]:
            return i
    return len(index_mapping)-1

def cepstral_mean_transform(mfcc):
    mean = np.mean(mfcc, axis=0)
    return mfcc - mean

class AudioDataset(Dataset):
    def add(self, data, context, total):
        length = len(data)
        padded = pad(data, context)
        self.mfccs.append(padded)
        return total + length

    def ingest_partition(self, total, root, phonemes, context, partition, augment):
        mfcc_dir = f"{root}/{partition}/mfcc"
        transcript_dir = f"{root}/{partition}/transcript"
        mfcc_names = sorted(os.listdir(mfcc_dir))
        transcript_names = sorted(os.listdir(transcript_dir))

        assert len(mfcc_names) == len(transcript_names)

        print(f"MFCCs in partition: {partition}")
        print(len(mfcc_names))

        for i in range(len(mfcc_names)):
            if i % 1000 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            mfcc = read_file(os.path.join(mfcc_dir, mfcc_names[i]))
            mfcc = cepstral_mean_transform(mfcc)

            transcript = read_file(os.path.join(transcript_dir, transcript_names[i]))
            t = np.copy(transcript[1:-1])

            self.index_mapping.append(total)
            total = self.add(mfcc, context, total)
            self.transcripts.append(t)

        return total

    def __init__(self, root, phonemes, context=30, partition="train-clean-100", augment=True):
        self.context = context
        self.phonemes = phonemes
        self.augment = augment

        self.mfccs = []
        self.transcripts = []
        self.index_mapping = []

        total = self.ingest_partition(0, root, phonemes, context, partition, augment)

        self.transcripts = np.concatenate(self.transcripts, axis=0)
        self.length = total
        self.transcripts = list(map(lambda ph: phonemes.index(ph), self.transcripts))

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        line = locate_line(self.index_mapping, ind)
        actual_index = (ind - self.index_mapping[line]) + MAX_CONTEXT

        before_context = self.context
        after_context = self.context

        lower_offset = actual_index - before_context
        upper_offset = actual_index + after_context + 1

        frames = self.mfccs[line]

        if self.augment:
            from .augment import pick_random_transform
            xform = pick_random_transform()
            frames = xform(frames, lower_offset, upper_offset)
        else:
            frames = frames[lower_offset:upper_offset]

        frames = frames.flatten()
        frames = torch.FloatTensor(frames)
        phonemes = torch.tensor(self.transcripts[ind])

        return frames, phonemes

class AudioTestDataset(Dataset):
    def __init__(self, root, phonemes, context=0, partition="test-clean"):
        self.context = context
        self.phonemes = phonemes

        self.mfcc_dir = f"{root}/{partition}/mfcc"
        mfcc_names = sorted(os.listdir(self.mfcc_dir))

        self.mfccs = []
        padding = np.zeros((self.context, 28))

        self.index_mapping = []
        total = 0

        for i in range(len(mfcc_names)):
            mfcc = read_file(os.path.join(self.mfcc_dir, mfcc_names[i]))
            mfcc = cepstral_mean_transform(mfcc)

            self.index_mapping.append(total)
            total = total + len(mfcc)

            mfcc = np.concatenate([np.copy(padding), mfcc, np.copy(padding)])
            self.mfccs.append(mfcc)

        self.length = total

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        line = locate_line(self.index_mapping, ind)
        actual_index = (ind - self.index_mapping[line]) + self.context

        before_context = self.context
        after_context = self.context

        lower_offset = actual_index - before_context
        upper_offset = actual_index + after_context + 1

        frames = self.mfccs[line][lower_offset:upper_offset]
        frames = frames.flatten()
        frames = torch.FloatTensor(frames)

        return frames