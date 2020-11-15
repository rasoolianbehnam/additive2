import os
import random
import time
from glob import glob
from os.path import basename, dirname, expanduser, join, splitext

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from pathos.multiprocessing import ProcessingPool, ThreadPool

DATA_DIR = "part_analysis_result/"
H5NAMES = ["April26_AMboneA1_volume", "April27_AMA2_volume"]
