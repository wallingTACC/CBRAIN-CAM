"""
Just all the imports for all other scripts and notebooks.
tgb - 2/7/2019 - Replacing keras with tensorflow.keras for eager execution purposes
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf
#import tensorflow.keras
#from tensorflow.keras.callbacks import *
from collections import OrderedDict
import pandas as pd
import pickle
import pdb
import netCDF4 as nc
import xarray as xr
import h5py
from glob import glob
import sys, os
import seaborn as sns
base_dir = os.getcwd().split('CBRAIN-CAM/')[0] + 'CBRAIN-CAM/'
sys.path.append(f'{base_dir}keras_network/')
sys.path.append(f'{base_dir}data_processing/')
#from .losses import *
#from .models import PartialReLU, QLayer, ELayer, MasConsLay, EntConsLay, SurRadLay
#from tensorflow.keras.utils import get_custom_objects
#metrics_dict = dict([(f.__name__, f) for f in all_metrics])
#get_custom_objects().update(metrics_dict)
# get_custom_objects().update({
#    'PartialReLU': PartialReLU,
#    'QLayer': QLayer,
#    'ELayer': ELayer,
#    'MasConsLay': MasConsLay,
#    'EntConsLay': EntConsLay,
#    'SurRadLay': SurRadLay
#    })
from os import path
from configargparse import ArgParser
#import fire
import logging
from ipykernel.kernelapp import IPKernelApp
def in_notebook():
    return IPKernelApp.initialized()

if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

#import tensorflow as tf
#from tensorflow.python.ops.parallel_for.gradients import jacobian

# def limit_mem():
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     tf.Session(config=config)

#limit_mem()



with open(os.path.join(os.path.dirname(__file__), 'hyai_hybi.pkl'), 'rb') as f:
    hyai, hybi = pickle.load(f)
