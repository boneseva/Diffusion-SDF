import time
import logging
import os
import random

import numpy as np
import torch
import torch.utils.data
import trimesh

from . import base
from tqdm import tqdm

import pandas as pd
import csv

# New VoxelLoader class
class VoxelLoader(torch.utils.data.Dataset):
    def __init__(self, voxel_paths):
        self.volumes = [np.load(p) for p in voxel_paths]  # Load binary voxels (B, D, D, D)
        print("Loading {} voxel files into memory...".format(len(self.volumes)))

    def __getitem__(self, idx):
        volume = torch.tensor(self.volumes[idx], dtype=torch.float32)
        return {'volume': volume, 'sdf': self._volume_to_sdf(volume)}  # Convert to SDF if needed

    def __len__(self):
        return len(self.volumes)

    def _volume_to_sdf(self, volume):
        # Optional: Convert binary voxel to SDF using EDT
        return torch.tensor(trimesh.voxel.grid_sdf(volume.numpy()))
