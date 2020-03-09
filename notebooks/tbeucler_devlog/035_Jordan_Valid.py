# tgb - 4/22/2019 - Use +1K as validation dataset
# tgb - 4/19/2019 - The goal is to make a slurm-callable script to calculate the statistics and residuals of all the paper neural networks over the validation dataset. This script is specialized to the +0K experiment.
# tgb - 1/6/2019 - Rerunning the statistics over the training set

import os
os.chdir('/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM')

from cbrain.imports import *
from cbrain.data_generator import *
from cbrain.cam_constants import *
from cbrain.losses import *
from cbrain.utils import limit_mem
from cbrain.layers import *
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
# Otherwise tensorflow will use ALL your GPU RAM for no reason
limit_mem()
TRAINDIR = '/local/Tom.Beucler/SPCAM_PHYS/'
DATADIR = '/project/meteo/w2w/A6/S.Rasp/SP-CAM/fluxbypass_aqua/'

config_fn = '/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM/pp_config/8col_rad_tbeucler_local_PostProc.yml'
data_fn = '/local/Tom.Beucler/SPCAM_PHYS/8col009_01_valid.nc'
dict_lay = {'SurRadLayer':SurRadLayer,'MassConsLayer':MassConsLayer,'EntConsLayer':EntConsLayer}

NNa = ['JNNL','JNNL0.01','JNNC','MLRL0']
for NNs in NNa:
    print('NNs = ',NNs)
    NN = {}; md = {};
    
    # 1) Load model
    path = TRAINDIR+'HDF5_DATA/'+NNs+'.h5'
    NN = load_model(path,custom_objects=dict_lay)
    
    # 2) Define model diagnostics object
    md = ModelDiagnostics(NN,config_fn,data_fn)
    
    # 3) Calculate statistics and save in pickle file
    md.compute_stats()
    path = TRAINDIR+'HDF5_DATA/'+NNs+'md_valid.pkl'
    pickle.dump(md.stats,open(path,'wb'))
    print('Stats are saved in ',path)
    
    # 4) Calculate budget residuals and save in pickle file
    md.compute_res()
    path = TRAINDIR+'HDF5_DATA/'+NNs+'res_valid.pkl'
    pickle.dump(md.res,open(path,'wb'))
    print('Budget residuals are saved in ',path)