from gym.wrappers import LazyFrames
import numpy as np
import torch


def compress_frames(obs):
    return LazyFrames(obs, lz4_compress=True)


def decompress_as_np(obs):
    return np.array(obs)


def decompress_frames(obs):
    return torch.tensor(np.array(obs))
