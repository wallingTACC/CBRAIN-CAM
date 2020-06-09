import sys
import os
DEV_DIR = '/work/00157/walling/projects/cloud_emulator/walling-CBRAIN-CAM/'
CONDA_DIR = '/work/00157/walling/conda/envs/CbrainCustomLayer/'
DATA_DIR = '/work/00157/walling/projects/cloud_emulator/walling-CBRAIN-CAM/notebooks/tbeucler_devlog/sherpa/data/'
SHERPA_TEMP_DIR = '/work/00157/walling/projects/cloud_emulator/walling-CBRAIN-CAM/notebooks/tbeucler_devlog/sherpa/sherpa-temp/'
EPOCHS = 10
#TRAINDIR = DATA_DIR

os.chdir(DEV_DIR)
sys.path.insert(1,CONDA_DIR + "lib/python3.7/site-packages") #work around for h5py
sys.path.insert(1, DEV_DIR)
sys.path.insert(1, DEV_DIR + "cbrain")
sys.path.insert(1, DEV_DIR + "notebooks/tbeucler_devlog")

import sherpa

client = sherpa.Client(host='127.0.0.1') #, port='37001')
trial = client.get_trial()

print(os.environ['SHERPA_RESOURCE'])
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SHERPA_RESOURCE']
                
from cbrain.imports import *
from cbrain.cam_constants import *
from cbrain.utils import *
from cbrain.layers import *
from cbrain.data_generator import DataGenerator
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow_probability as tfp
import xarray as xr
import numpy as np
from cbrain.model_diagnostics import ModelDiagnostics
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as imag
import scipy.integrate as sin
# import cartopy.crs as ccrs
import matplotlib.ticker as mticker
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
# from climate_invariant import *
from tensorflow.keras import layers
import datetime
from climate_invariant_utils import *
import yaml

tf.debugging.set_log_device_placement(True)

# ## Global Variables

# In[3]:
#TRAINDIR = '/oasis/scratch/comet/ankitesh/temp_project/PrepData/CRHData/'
data_path = '/work/00157/walling/projects/cloud_emulator/walling-CBRAIN-CAM/notebooks/tbeucler_devlog/sherpa/data/'


# Load coordinates (just pick any file from the climate model run)
coor = xr.open_dataset(data_path + "sp8fbp_minus4k.cam2.h1.0000-01-01-00000.nc", decode_times=False)
lat = coor.lat; lon = coor.lon; lev = coor.lev;
coor.close();

# Load hyam and hybm to calculate pressure field in SPCAM
PATH = DEV_DIR + 'cbrain/'
path_hyam = 'hyam_hybm.pkl'
hf = open(path_hyam,'rb')
hyam,hybm = pickle.load(hf)

# Scale dictionary to convert the loss to W/m2
scale_dict = load_pickle(data_path + '009_Wm2_scaling_2.pkl')


# New Data generator class for the climate-invariant network. Calculates the physical rescalings needed to make the NN climate-invariant

# In[5]:


class DataGeneratorClimInv(DataGenerator):
    
    def __init__(self, data_fn, input_vars, output_vars,
             norm_fn=None, input_transform=None, output_transform=None,
             batch_size=trial.parameters['batch_size'], shuffle=True, xarray=False, var_cut_off=None,
             rh_trans=True,t2tns_trans=True,
             lhflx_trans=True,
             scaling=True,interpolate=True,
             hyam=None,hybm=None,                 
             inp_subRH=None,inp_divRH=None,
             inp_subTNS=None,inp_divTNS=None,
             lev=None, interm_size=40,
             lower_lim=6,
             is_continous=True,Tnot=5,
                mode='train'):
        
        self.scaling = scaling
        self.interpolate = interpolate
        self.rh_trans = rh_trans
        self.t2tns_trans = t2tns_trans
        self.lhflx_trans = lhflx_trans
        self.inp_shape = 64
        self.mode=mode
        super().__init__(data_fn, input_vars,output_vars,norm_fn,input_transform,output_transform,
                        batch_size,shuffle,xarray,var_cut_off) ## call the base data generator
        self.inp_sub = self.input_transform.sub
        self.inp_div = self.input_transform.div
        if rh_trans:
            self.qv2rhLayer = QV2RHNumpy(self.inp_sub,self.inp_div,inp_subRH,inp_divRH,hyam,hybm)
        
        if lhflx_trans:
            self.lhflxLayer = LhflxTransNumpy(self.inp_sub,self.inp_div,hyam,hybm)
            
        if t2tns_trans:
            self.t2tnsLayer = T2TmTNSNumpy(self.inp_sub,self.inp_div,inp_subTNS,inp_divTNS,hyam,hybm)
            
        if scaling:
            self.scalingLayer = ScalingNumpy(hyam,hybm)
            self.inp_shape += 1
                    
        if interpolate:
            self.interpLayer = InterpolationNumpy(lev,is_continous,Tnot,lower_lim,interm_size)
            self.inp_shape += interm_size*2 + 4 + 30 ## 4 same as 60-64 and 30 for lev_tilde.size
        
            
        
    def __getitem__(self, index):
        # Compute start and end indices for batch
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        # Grab batch from data
        batch = self.data_ds['vars'][start_idx:end_idx]

        # Split into inputs and outputs
        X = batch[:, self.input_idxs]
        Y = batch[:, self.output_idxs]
        # Normalize
        X_norm = self.input_transform.transform(X)
        Y = self.output_transform.transform(Y)
        X_result = X_norm
        
        if self.rh_trans:
            X_result = self.qv2rhLayer.process(X_result) 
            
        if self.lhflx_trans:
            X_result = self.lhflxLayer.process(X_result)
            X_result = X_result[:,:64]
            X = X[:,:64]
            
        if self.t2tns_trans:
            X_result = self.t2tnsLayer.process(X_result)
        
        if self.scaling:
            scalings = self.scalingLayer.process(X) 
            X_result = np.hstack((X_result,scalings))
        
        if self.interpolate:
            interpolated = self.interpLayer.process(X,X_result)
            X_result = np.hstack((X_result,interpolated))
            

        if self.mode=='val':
            return xr.DataArray(X_result), xr.DataArray(Y)
        return X_result,Y
    
    ##transforms the input data into the required format, take the unnormalized dataset
    def transform(self,X):
        X_norm = self.input_transform.transform(X)
        X_result = X_norm
        
        if self.rh_trans:
            X_result = self.qv2rhLayer.process(X_result)  
        
        if self.lhflx_trans:
            X_result = self.lhflxLayer.process(X_result)
            X_result = X_result[:,:64]
            X = X[:,:64]

        if self.t2tns_trans:
            X_result = self.t2tnsLayer.process(X_result)
        
        if self.scaling:
            scalings = self.scalingLayer.process(X) 
            X_result = np.hstack((X_result,scalings))
        
        if self.interpolate:
            interpolated = self.interpLayer.process(X,X_result)
            X_result = np.hstack((X_result,interpolated))
            

        return X_result


# ## Data Generators

# ### Choose between aquaplanet and realistic geography here

# In[6]:


path_aquaplanet = data_path + 'phase1/aquaplanet/'
path_realgeography = data_path + 'phase1/geography/'

#path = path_aquaplanet 
#out_vars_RH = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']

path = path_realgeography
out_vars_RH = ['PTEQ','PTTEND','FSNT', 'FSNS', 'FLNT', 'FLNS']

out_vars = out_vars_RH



# ### Data Generator using RH

# In[7]:


scale_dict_RH = load_pickle(data_path + '009_Wm2_scaling_2.pkl')
scale_dict_RH['RH'] = 0.01*L_S/G, # Arbitrary 0.1 factor as specific humidity is generally below 2%

in_vars_RH = ['RH','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
#out_vars_RH = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']

TRAINFILE_RH = 'CI_RH_M4K_NORM_train_shuffle.nc'
NORMFILE_RH = 'CI_RH_M4K_NORM_norm.nc'
#VALIDFILE_RH = 'CI_RH_M4K_NORM_valid.nc' # Experiment 1/2
VALIDFILE_RH = 'CI_RH_P4K_NORM_valid.nc' # Experiment 3/4


# In[8]:


train_gen_RH = DataGenerator(
    data_fn = path+TRAINFILE_RH,
    input_vars = in_vars_RH,
    output_vars = out_vars_RH,
    norm_fn = path+NORMFILE_RH,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_RH,
    batch_size=trial.parameters['batch_size'],
    shuffle=True,
)


# ### Data Generator using TNS

# In[9]:


in_vars = ['QBP','TfromNS','PS', 'SOLIN', 'SHFLX', 'LHFLX']
#out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']

TRAINFILE_TNS = 'CI_TNS_M4K_NORM_train_shuffle.nc'
NORMFILE_TNS = 'CI_TNS_M4K_NORM_norm.nc'
#VALIDFILE_TNS = 'CI_TNS_M4K_NORM_valid.nc' # Experiment 1/2
VALIDFILE_TNS = 'CI_TNS_P4K_NORM_valid.nc' # Experiment 3/4


# In[10]:


train_gen_TNS = DataGenerator(
    data_fn = path+TRAINFILE_TNS,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = path+NORMFILE_TNS,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=trial.parameters['batch_size'],
    shuffle=True,
)


# ## Data Generator Combined

# In[11]:


in_vars = ['QBP','TBP','PS', 'SOLIN', 'SHFLX', 'LHFLX']
#out_vars = ['PHQ','TPHYSTND','FSNT', 'FSNS', 'FLNT', 'FLNS']


# ## Brute-Force Model

# In[12]:


TRAINFILE = 'CI_SP_M4K_train_shuffle.nc'
NORMFILE = 'CI_SP_M4K_NORM_norm.nc'
#VALIDFILE = 'CI_SP_M4K_valid.nc' # Experiment 1/2
VALIDFILE = 'CI_SP_P4K_valid.nc' # Experiment 3/4

print('Batch Size = ' + str(trial.parameters['batch_size']))
print('Num Layers = ' + str(trial.parameters['num_layers']))
print('Num Units = ' + str(trial.parameters['num_units']))

train_gen = DataGeneratorClimInv(
    data_fn = path+TRAINFILE,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = path+NORMFILE,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=trial.parameters['batch_size'],
    shuffle=True,
    lev=lev,
    hyam=hyam,hybm=hybm,
    inp_subRH=train_gen_RH.input_transform.sub, inp_divRH=train_gen_RH.input_transform.div,
    inp_subTNS=train_gen_TNS.input_transform.sub,inp_divTNS=train_gen_TNS.input_transform.div,
    rh_trans=False,t2tns_trans=False,
    lhflx_trans=False,
    scaling=False,
    interpolate=False
)

valid_gen = DataGeneratorClimInv(
    data_fn = path+VALIDFILE,
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = path+NORMFILE,
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict,
    batch_size=trial.parameters['batch_size'],
    shuffle=True,
    lev=lev,
    hyam=hyam,hybm=hybm,
    inp_subRH=train_gen_RH.input_transform.sub, inp_divRH=train_gen_RH.input_transform.div,
    inp_subTNS=train_gen_TNS.input_transform.sub,inp_divTNS=train_gen_TNS.input_transform.div,
    rh_trans=False,t2tns_trans=False,
    lhflx_trans=False,
    scaling=False,
    interpolate=False
)

inp = Input(shape=(64,)) ## input after rh and tns transformation
#densout = Dense(128, activation='linear')(inp)
densout = Dense(trial.parameters['num_units'], activation='linear')(inp)

densout = LeakyReLU(alpha=0.3)(densout)
for i in range(trial.parameters['num_layers']): #range (6):
    #densout = Dense(128, activation='linear')(densout)
    densout = Dense(trial.parameters['num_units'], activation='linear')(densout)
    densout = LeakyReLU(alpha=0.3)(densout)
dense_out = Dense(64, activation='linear')(densout)
model = tf.keras.models.Model(inp, dense_out)
model.compile(tf.keras.optimizers.Adam(), loss=mse)
Nep = 10
#model.summary()
model.fit_generator(train_gen, epochs=Nep, validation_data=valid_gen, 
                    callbacks=[client.keras_send_metrics(trial,
                                                   objective_name='val_loss',
                                                   context_names=['loss'])
                              ])

