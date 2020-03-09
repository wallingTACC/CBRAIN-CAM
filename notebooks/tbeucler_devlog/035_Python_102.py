# tgb - 2/7/2020 - Calculating residuals for the three test networks of notebook 035
# tgb - 8/9/2019 - Calculating statistics for all networks from the sensitivity tests on +0,4K datasets
# tgb - 7/3/2019 - Calculates statistics for JUnotC network on +0K and +4K datets
# tgb - 6/21/2019 - Evaluates Jordan networks on the Wavenumber 1-forced +3K dataset
# tgb - 6/3/2019 - Calculates statistics on all of Jordan networks
# tgb - 5/1/2019 - Calculates statistics on NNLA for all datasets
# tgb - 4/27/2019 - Calculates precipitation PDF for each network on +0K and +4K
# tgb - 4/24/2019 - Data-scarce using just 1-file --> calculate statistics on +0,1,2,3,4K; NNA version
# tgb - 4/24/2019 - Validate the unconstrained multiple linear regression model on +1,2,3,4K
# tgb - 4/22/2019 - Use +1K as validation dataset
# tgb - 4/19/2019 - The goal is to make a slurm-callable script to calculate the statistics and residuals of all the paper neural networks over the validation dataset. This script is specialized to the +0K experiment.

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

path = '/home/t/Tom.Beucler/SPCAM/CBRAIN-CAM/cbrain/'
path_hyai = 'hyam_hybm.pkl'
hf = open(path+path_hyai,'rb')
test = pickle.load(hf)

# Moist thermodynamics library adapted to tf
def eliq(T):
    a_liq = np.float32(np.array([-0.976195544e-15,-0.952447341e-13,\
                                 0.640689451e-10,\
                      0.206739458e-7,0.302950461e-5,0.264847430e-3,\
                      0.142986287e-1,0.443987641,6.11239921]));
    c_liq = np.float32(-80.0)
    T0 = np.float32(273.16)
    return np.float32(100.0)*tfm.polyval(a_liq,tfm.maximum(c_liq,T-T0))

def eice(T):
    a_ice = np.float32(np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,\
                      0.602588177e-7,0.615021634e-5,0.420895665e-3,\
                      0.188439774e-1,0.503160820,6.11147274]));
    c_ice = np.float32(np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07]))
    T0 = np.float32(273.16)
    return tf.where(T>c_ice[0],eliq(T),\
                   tf.where(T<=c_ice[1],np.float32(100.0)*(c_ice[3]+tfm.maximum(c_ice[2],T-T0)*\
                   (c_ice[4]+tfm.maximum(c_ice[2],T-T0)*c_ice[5])),\
                           np.float32(100.0)*tfm.polyval(a_ice,T-T0)))

def esat(T):
    T0 = np.float32(273.16)
    T00 = np.float32(253.16)
    omtmp = (T-T00)/(T0-T00)
    omega = tfm.maximum(np.float32(0.0),tfm.minimum(np.float32(1.0),omtmp))

    return tf.where(T>T0,eliq(T),tf.where(T<T00,eice(T),(omega*eliq(T)+(1-omega)*eice(T))))

def qv(T,RH,P0,PS,hyam,hybm):
    
    R = np.float32(287.0)
    Rv = np.float32(461.0)
    p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)
    
    T = tf.cast(T,tf.float32)
    RH = tf.cast(RH,tf.float32)
    p = tf.cast(p,tf.float32)
    
    return R*esat(T)*RH/(Rv*p)
    # DEBUG 1
    # return esat(T)
    
def RH(T,qv,P0,PS,hyam,hybm):
    R = np.float32(287.0)
    Rv = np.float32(461.0)
    p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)
    
    T = tf.cast(T,tf.float32)
    qv = tf.cast(qv,tf.float32)
    p = tf.cast(p,tf.float32)
    
    return Rv*p*qv/(R*esat(T))

class RH2QV(Layer):
    def __init__(self, inp_subQ, inp_divQ, inp_subRH, inp_divRH, hyam, hybm, **kwargs):
        """
        Call using ([input])
        Assumes
        prior: [RHBP, 
        QCBP, QIBP, TBP, VBP, Qdt_adiabatic, QCdt_adiabatic, QIdt_adiabatic, 
        Tdt_adiabatic, Vdt_adiabatic, PS, SOLIN, SHFLX, LHFLX]
        Returns
        post(erior): [QBP, 
        QCBP, QIBP, TBP, VBP, Qdt_adiabatic, QCdt_adiabatic, QIdt_adiabatic, 
        Tdt_adiabatic, Vdt_adiabatic, PS, SOLIN, SHFLX, LHFLX]
        """
        self.inp_subQ, self.inp_divQ, self.inp_subRH, self.inp_divRH, self.hyam, self.hybm = \
            np.array(inp_subQ), np.array(inp_divQ), np.array(inp_subRH), np.array(inp_divRH), \
        np.array(hyam), np.array(hybm)
        # Define variable indices here
        # Input
        self.QBP_idx = slice(0,30)
        self.TBP_idx = slice(90,120)
        self.PS_idx = 300
        self.SHFLX_idx = 302
        self.LHFLX_idx = 303

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_subQ': list(self.inp_subQ), 'inp_divQ': list(self.inp_divQ),
                  'inp_subRH': list(self.inp_subRH), 'inp_divRH': list(self.inp_divRH),
                  'hyam': list(self.hyam),'hybm': list(self.hybm)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, arrs):
        prior = arrs
        
        Tprior = prior[:,self.TBP_idx]*self.inp_divRH[self.TBP_idx]+self.inp_subRH[self.TBP_idx]
        RHprior = prior[:,self.QBP_idx]*self.inp_divRH[self.QBP_idx]+self.inp_subRH[self.QBP_idx]
        PSprior = prior[:,self.PS_idx]*self.inp_divRH[self.PS_idx]+self.inp_subRH[self.PS_idx]
        qvprior = (qv(Tprior,RHprior,P0,PSprior,self.hyam,self.hybm)-\
                    self.inp_subQ[self.QBP_idx])/self.inp_divQ[self.QBP_idx]
        
        post = tf.concat([tf.cast(qvprior,tf.float32),prior[:,30:]], axis=1)
        
        return post

    def compute_output_shape(self,input_shape):
        """Input shape + 1"""
        return (input_shape[0][0])
    
class dQVdt2dRHdt(Layer):
    def __init__(self, inp_subQ, inp_divQ, norm_qQ, inp_subRH, inp_divRH, norm_qRH, hyam, hybm, **kwargs):
        """
        Call using ([input_qv,output])
        Assumes
        prior: [PHQ, PHCLDLIQ, PHCLDICE, TPHYSTND, QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        Returns
        post(erior): [dRHdt, PHCLDLIQ, PHCLDICE, TPHYSTND, QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        """
        self.inp_subQ, self.inp_divQ, self.norm_qQ, \
        self.inp_subRH, self.inp_divRH, self.norm_qRH, \
        self.hyam, self.hybm = \
            np.array(inp_subQ), np.array(inp_divQ), np.array(norm_qQ), \
        np.array(inp_subRH), np.array(inp_divRH), np.array(norm_qRH), \
        np.array(hyam), np.array(hybm)
        # Define variable indices here
        # Input
        self.PHQ_idx = slice(0,30)
        
        self.QBP_idx = slice(0,30)
        self.TBP_idx = slice(90,120)
        self.PS_idx = 300
        self.SHFLX_idx = 302
        self.LHFLX_idx = 303

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_subQ': list(self.inp_subQ), 'inp_divQ': list(self.inp_divQ),
                  'norm_qQ': list(self.norm_qQ),
                  'inp_subRH': list(self.inp_subRH), 'inp_divRH': list(self.inp_divRH),
                  'norm_qRH': list(self.norm_qRH), 
                  'hyam': list(self.hyam),'hybm': list(self.hybm)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, arrs):
        inp, prior = arrs
        
        # Assumes the input has specific humidity in positions[0,30]
        Tprior = inp[:,self.TBP_idx]*self.inp_divQ[self.TBP_idx]+self.inp_subQ[self.TBP_idx]
        PSprior = inp[:,self.PS_idx]*self.inp_divQ[self.PS_idx]+self.inp_subQ[self.PS_idx]
        dqvdtprior = prior[:,self.QBP_idx]/self.norm_qQ
        
        dRHdtprior = RH(Tprior,dqvdtprior,P0,PSprior,self.hyam,self.hybm)*self.norm_qRH
        post = tf.concat([dRHdtprior,prior[:,30:]], axis=1)
        
        return post

    def compute_output_shape(self,input_shape):
        """Input shape"""
        return (input_shape[0][0],input_shape[0][1])

path_HDF5 = '/local/Tom.Beucler/SPCAM_PHYS/HDF5_DATA/'
config_file = ['8col_rad_tbeucler_local-RH_PostProc.yml',
               '8col_rad_tbeucler_local-RH_PostProc.yml',
               '8col_rad_tbeucler_local-RH_PostProc.yml']
data_file = ['8col009RH_01_train.nc',
            '8col009RH_01_valid.nc',
            '8col009RH_01_test.nc']
NNarray = ['035_Test01.hdf5','035_Test02.hdf5','035_Test03.hdf5']
NNname = ['UCnet','UCnet_{NL}','ACnet_{NL}']
dict_lay = {'SurRadLayer':SurRadLayer,'MassConsLayer':MassConsLayer,'EntConsLayer':EntConsLayer,
            'RH2QV':RH2QV,'dQVdt2dRHdt':dQVdt2dRHdt,
           'eliq':eliq,'eice':eice,'esat':esat,'qv':qv,'RH':RH}

NN = {}; md = {};
os.chdir('/local/Tom.Beucler/SPCAM_PHYS/HDF5_DATA')

for i,NNs in enumerate(NNarray):
    print('NN name is ',NNs)
    path = path_HDF5+NNs
    NN[NNs] = load_model(path,custom_objects=dict_lay)
    md[NNs] = {}
    for j,data in enumerate(data_file):
        print('data name is ',data)
        md[NNs][data[13:-3]] = ModelDiagnostics(NN[NNs],
                                                '/home/t/Tom.Beucler/SPCAM/CBRAIN-CAM/pp_config/'+config_file[i],
                                                '/local/Tom.Beucler/SPCAM_PHYS/'+data)
        md[NNs][data[13:-3]].compute_res()
        pickle.dump(md.res,open(TRAINDIR+'HDF5_DATA/'+NNs+'mdres'+dataref[j]+'.pkl','wb'))
