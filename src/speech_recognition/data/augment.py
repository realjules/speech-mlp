import random
import numpy as np

def time_shift(frames, lower_offset, upper_offset, shift_range=(-25, 25)):
    """Time shift augmentation"""
    shift = random.randint(*shift_range)
    return frames[lower_offset+shift:upper_offset+shift]

def freq_mask(frames, lower_offset, upper_offset, max_width=8):
    """Frequency masking augmentation"""
    frames = frames[lower_offset:upper_offset]
    freq_width = random.randint(1, max_width)
    freq_start = random.randint(0, frames.shape[1] - freq_width)
    frames[:, freq_start:freq_start+freq_width] = 0
    return frames

def time_mask(frames, lower_offset, upper_offset, max_width=20):
    """Time masking augmentation"""
    frames = frames[lower_offset:upper_offset]
    time_width = random.randint(1, max_width)
    time_start = random.randint(0, frames.shape[0] - time_width)
    frames[time_start:time_start+time_width, :] = 0
    return frames

def identity(frames, lower_offset, upper_offset):
    """No augmentation"""
    return frames[lower_offset:upper_offset]

def pick_random_transform():
    """Randomly select an augmentation transform"""
    transforms = [
        time_shift,
        freq_mask,
        time_mask,
        identity
    ]
    return random.choice(transforms)