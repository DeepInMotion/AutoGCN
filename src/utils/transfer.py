import os
from shutil import rmtree, copy
import numpy as np
import argparse

transfer_settings = {
    'kinetics': {
        'remove_humans': [1],  # Keep only the first person found in a video
        'remove_channels': [2],  # Remove confidence
        'remove_joints': [14, 15],  # REye, LEye
        'permutation': [],  # New order which fits the openpose-in-motion pose
    },
    'ntu': {
        'remove_humans': [1],
        'remove_channels': [2],
        'remove_joints': [1, 7, 11, 15, 19, 21, 22, 23, 24],    # spine-middle, lhand, rhand, lfoot, rfoot, tip-lhand,
                                                                # lthumb, tip-rhand, rthumb
        'permutation': []
    }
}

