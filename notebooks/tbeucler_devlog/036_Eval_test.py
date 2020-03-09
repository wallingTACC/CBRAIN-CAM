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
PREFIX = '8col009_01_'

PREFIXDS = PREFIX
os.chdir('/filer/z-sv-pool12c/t/Tom.Beucler/SPCAM/CBRAIN-CAM')

class MassConsLayer_choice(Layer):
    def __init__(self, inp_sub, inp_div, norm_q, hyai, hybi, lvl_choice, **kwargs):
        """
        Call using ([input, output])
        Assumes
        prior: [PHQ_nores, PHCLDLIQ, PHCLDICE, TPHYSTND_nores,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        Returns
        post(erior): [PHQ, PHCLDLIQ, PHCLDICE, TPHYSTND_nores,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        Added lvl_choice, a hyper-parameter to choose the level of mass conservation [0-29]
        """
        self.inp_sub, self.inp_div, self.norm_q, self.hyai, self.hybi = \
            np.array(inp_sub), np.array(inp_div), np.array(norm_q), np.array(hyai), np.array(hybi)
        self.lvl_choice = np.int32(lvl_choice)
        # Define variable indices here
        # Input
        self.PS_idx = 300
        self.LHFLX_idx = 303
        # Output
        self.PHQbef_idx = slice(0, self.lvl_choice) # Indices before the residual
        self.PHCLDLIQ_idx = slice(29, 59)
        self.PHCLDICE_idx = slice(59, 89)
        self.PRECT_idx = 212
        self.PRECTEND_idx = 213

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_sub': list(self.inp_sub), 'inp_div': list(self.inp_div),
                  'norm_q': list(self.norm_q), 'hyai': list(self.hyai),
                  'hybi': list(self.hybi), 'lvl_choice':self.lvl_choice}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, arrs):
        inp, prior = arrs

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(
            inp[:, self.PS_idx],
            self.inp_div[self.PS_idx], self.inp_sub[self.PS_idx],
            self.norm_q, self.hyai, self.hybi
        )

        # 2. Compute vertical cloud water integral
        CLDINT = K.sum(dP_tilde *
                       (prior[:, self.PHCLDLIQ_idx] + prior[:, self.PHCLDICE_idx]),
                       axis=1)

        # 3. Compute water vapor integral minus the water vapor residual
        # Careful with handling the pressure vector since it is not aligned
        # with the prior water vapor vector
        VAPINT = K.sum(dP_tilde[:, self.PHQbef_idx] * prior[:, self.PHQbef_idx], 1) +\
        K.sum(dP_tilde[:, self.lvl_choice+1:30] * prior[:, self.lvl_choice:29], 1)

        # 4. Compute forcing (see Tom's note for details, I am just copying)
        LHFLX = (inp[:, self.LHFLX_idx] * self.inp_div[self.LHFLX_idx] +
                 self.inp_sub[self.LHFLX_idx])
        PREC = prior[:, self.PRECT_idx] + prior[:, self.PRECTEND_idx]

        # 5. Compute water vapor tendency at level lvl_choice as residual
        PHQ_LVL = (LHFLX - PREC - CLDINT - VAPINT) / dP_tilde[:, self.lvl_choice]

        # 6. Concatenate output vector
        post = tf.concat([
            prior[:, self.PHQbef_idx], PHQ_LVL[:, None],
            prior[:, self.lvl_choice:]
        ], axis=1)
        return post

    def compute_output_shape(self, input_shape):
        """Input shape + 1"""
        return (input_shape[0][0], input_shape[0][1] + 1)
    
class EntConsLayer_choice(Layer):
    def __init__(self, inp_sub, inp_div, norm_q, hyai, hybi, lvl_choice, **kwargs):
        """
        Call using ([input, output])
        Assumes
        prior: [PHQ, PHCLDLIQ, PHCLDICE, TPHYSTND_nores,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        Returns
        post(erior): [PHQ, PHCLDLIQ, PHCLDICE, TPHYSTND,
        QRL, QRS, DTVKE, FSNT, FSNS, FLNT, FLNS, PRECT, PRECTEND, PRECST, PRECSTEN]
        """
        self.inp_sub, self.inp_div, self.norm_q, self.hyai, self.hybi = \
            np.array(inp_sub), np.array(inp_div), np.array(norm_q), np.array(hyai), np.array(hybi)
        self.lvl_choice = np.int32(lvl_choice)
        # Define variable indices here
        # Input
        self.PS_idx = 300
        self.SHFLX_idx = 302
        self.LHFLX_idx = 303

        # Output
        self.PHQ_idx = slice(0, 30)
        self.PHCLDLIQ_idx = slice(30, 60)
        self.Tbef_idx = slice(90, 90+self.lvl_choice)
        self.DTVKE_idx = slice(179, 209)
        self.FSNT_idx = 209
        self.FSNS_idx = 210
        self.FLNT_idx = 211
        self.FLNS_idx = 212
        self.PRECT_idx = 213
        self.PRECTEND_idx = 214
        self.PRECST_idx = 215
        self.PRECSTEND_idx = 216

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_sub': list(self.inp_sub), 'inp_div': list(self.inp_div),
                  'norm_q': list(self.norm_q), 'hyai': list(self.hyai),
                  'hybi': list(self.hybi), 'lvl_choice': self.lvl_choice}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, arrs):
        inp, prior = arrs

        # 1. Compute dP_tilde
        dP_tilde = compute_dP_tilde(
            inp[:, self.PS_idx],
            self.inp_div[self.PS_idx], self.inp_sub[self.PS_idx],
            self.norm_q, self.hyai, self.hybi
        )

        # 2. Compute net energy input from phase change and precipitation
        PHAS = L_I / L_V * (
                (prior[:, self.PRECST_idx] + prior[:, self.PRECSTEND_idx]) -
                (prior[:, self.PRECT_idx] + prior[:, self.PRECTEND_idx])
        )

        # 3. Compute net energy input from radiation, SHFLX and TKE
        RAD = (prior[:, self.FSNT_idx] - prior[:, self.FSNS_idx] -
               prior[:, self.FLNT_idx] + prior[:, self.FLNS_idx])
        SHFLX = (inp[:, self.SHFLX_idx] * self.inp_div[self.SHFLX_idx] +
                 self.inp_sub[self.SHFLX_idx])
        KEDINT = K.sum(dP_tilde * prior[:, self.DTVKE_idx], 1)

        # 4. Compute tendency of vapor due to phase change
        LHFLX = (inp[:, self.LHFLX_idx] * self.inp_div[self.LHFLX_idx] +
                 self.inp_sub[self.LHFLX_idx])
        VAPINT = K.sum(dP_tilde * prior[:, self.PHQ_idx], 1)
        SPDQINT = (VAPINT - LHFLX) * L_S / L_V

        # 5. Same for cloud liquid water tendency
        SPDQCINT = K.sum(dP_tilde * prior[:, self.PHCLDLIQ_idx], 1) * L_I / L_V

        # 6. And the same for T but remember residual is still missing
        DTINT = K.sum(dP_tilde[:, :self.lvl_choice] *\
                      prior[:, self.Tbef_idx], 1) +\
        K.sum(dP_tilde[:, self.lvl_choice+1:30] *\
             prior[:, 90+self.lvl_choice:119], 1)

        # 7. Compute DT30 as residual
        DT_LVL = (
                       PHAS + RAD + SHFLX + KEDINT - SPDQINT - SPDQCINT - DTINT
               ) / dP_tilde[:, self.lvl_choice]

        # 8. Concatenate output vector
        post = tf.concat([
            prior[:, :(90+self.lvl_choice)], DT_LVL[:, None], \
            prior[:, (90+self.lvl_choice):]
        ], axis=1)
        return post

    def compute_output_shape(self, input_shape):
        """Input shape + 1"""
        return (input_shape[0][0], input_shape[0][1] + 1)
    
scale_dict = load_pickle('./nn_config/scale_dicts/009_Wm2_scaling.pkl')
in_vars = load_pickle('./nn_config/scale_dicts/009_Wm2_in_vars.pkl')
out_vars = load_pickle('./nn_config/scale_dicts/009_Wm2_out_vars.pkl')
dP = load_pickle('./nn_config/scale_dicts/009_Wm2_dP.pkl')

# Define levels to calculate as residuals
mlev = 14
elev = 14

multiplier = 1
scale_dict_mult = scale_dict
scale_dict_mult['PHQ'][mlev] *= multiplier
scale_dict_mult['TPHYSTND'][elev] *= multiplier

train_gen = DataGenerator(
    data_fn = TRAINDIR+PREFIXDS+'train_shuffle.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+PREFIX+'norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_mult,
    batch_size=1024,
    shuffle=True
)
valid_gen = DataGenerator(
    data_fn = TRAINDIR+PREFIX+'valid.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+PREFIX+'norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_mult,
    batch_size=1024,
    shuffle=False
)
test_gen = DataGenerator(
    data_fn = TRAINDIR+PREFIX+'test.nc',
    input_vars = in_vars,
    output_vars = out_vars,
    norm_fn = TRAINDIR+PREFIX+'norm.nc',
    input_transform = ('mean', 'maxrs'),
    output_transform = scale_dict_mult,
    batch_size=1024,
    shuffle=False
)

class Weighted_Loss():
    def __init__(self, lev1, lev2, multiplier, name='weighted_loss',**kwargs):
        self.lev1 = lev1
        self.lev2 = lev2
        self.multiplier = multiplier
        self.__name__ = name
        super().__init__(**kwargs)
    def __call__(self,ytrue,ypred):
        # Multiply two levels of y_pred by multiplier
        ypred_mult = tf.concat([
            ypred[:,:self.lev1],
            tf.expand_dims(self.multiplier*ypred[:,self.lev1],axis=1),
            ypred[:,self.lev1:self.lev2],
            tf.expand_dims(self.multiplier*ypred[:,self.lev2],axis=1),
            ypred[:,self.lev2:]
        ], axis=1)
        # Multiply two levels of y_truth by multiplier
        ytrue_mult = tf.concat([
            ytrue[:,:self.lev1],
            tf.expand_dims(self.multiplier*ytrue[:,self.lev1],axis=1),
            ytrue[:,self.lev1:self.lev2],
            tf.expand_dims(self.multiplier*ytrue[:,self.lev2],axis=1),
            ytrue[:,self.lev2:]
        ], axis=1)
        return (ypred_mult-ytrue_mult)**2
    
def Conserving_Model(maslevel,entlevel):
    inpC = Input(shape=(304,))
    densout = Dense(512, activation='linear')(inpC)
    densout = LeakyReLU(alpha=0.3)(densout)
    for i in range (4):
        densout = Dense(512, activation='linear')(densout)
        densout = LeakyReLU(alpha=0.3)(densout)
    densout = Dense(214, activation='linear')(densout)
    densout = LeakyReLU(alpha=0.3)(densout)
    surfout = SurRadLayer(
        inp_div=train_gen.input_transform.div,
        inp_sub=train_gen.input_transform.sub,
        norm_q=scale_dict['PHQ'],
        hyai=hyai, hybi=hybi
    )([inpC, densout])
    massout = MassConsLayer_choice(
        inp_div=train_gen.input_transform.div,
        inp_sub=train_gen.input_transform.sub,
        norm_q=scale_dict['PHQ'],
        hyai=hyai, hybi=hybi\
        , lvl_choice=maslevel
    )([inpC, surfout])
    enthout = EntConsLayer_choice(
        inp_div=train_gen.input_transform.div,
        inp_sub=train_gen.input_transform.sub,
        norm_q=scale_dict['PHQ'],
        hyai=hyai, hybi=hybi\
        , lvl_choice=entlevel
    )([inpC, massout])
    return tf.keras.models.Model(inpC, enthout)

def mass_res_diagno(inp_div,inp_sub,norm_q,inp,pred):
    # Input
    PS_idx = 300
    LHFLX_idx = 303

    # Output
    PHQ_idx = slice(0, 30)
    PHCLDLIQ_idx = slice(30, 60)
    PHCLDICE_idx = slice(60, 90)
    PRECT_idx = 214
    PRECTEND_idx = 215

    # 1. Compute dP_tilde
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

    # 2. Compute water integral
    WATINT = np.sum(dP_tilde *(pred[:, PHQ_idx] + pred[:, PHCLDLIQ_idx] + pred[:, PHCLDICE_idx]), axis=1)
#     print('PHQ',np.mean(np.sum(dP_tilde*pred[:,PHQ_idx],axis=1)))
#     print('PHCLQ',np.mean(np.sum(dP_tilde*pred[:,PHCLDLIQ_idx],axis=1)))
#     print('PHICE',np.mean(np.sum(dP_tilde*pred[:,PHCLDICE_idx],axis=1)))

    # 3. Compute latent heat flux and precipitation forcings
    LHFLX = inp[:, LHFLX_idx] * inp_div[LHFLX_idx] + inp_sub[LHFLX_idx]
    PREC = pred[:, PRECT_idx] + pred[:, PRECTEND_idx]

    # 4. Compute water mass residual
#     print('LHFLX',np.mean(LHFLX))
#     print('PREC',np.mean(PREC))
#     print('WATINT',np.mean(WATINT))
    WATRES = LHFLX - PREC - WATINT
    #print('WATRES',np.mean(WATRES))

    return np.square(WATRES)

def ent_res_diagno(inp_div,inp_sub,norm_q,inp,pred):

    # Input
    PS_idx = 300
    SHFLX_idx = 302
    LHFLX_idx = 303

    # Output
    PHQ_idx = slice(0, 30)
    PHCLDLIQ_idx = slice(30, 60)
    PHCLDICE_idx = slice(60, 90)
    TPHYSTND_idx = slice(90, 120)
    DTVKE_idx = slice(180, 210)
    FSNT_idx = 210
    FSNS_idx = 211
    FLNT_idx = 212
    FLNS_idx = 213
    PRECT_idx = 214
    PRECTEND_idx = 215
    PRECST_idx = 216
    PRECSTEND_idx = 217

    # 1. Compute dP_tilde
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

    # 2. Compute net energy input from phase change and precipitation
    PHAS = L_I / L_V * (
            (pred[:, PRECST_idx] + pred[:, PRECSTEND_idx]) -
            (pred[:, PRECT_idx] + pred[:, PRECTEND_idx])
    )

    # 3. Compute net energy input from radiation, SHFLX and TKE
    RAD = (pred[:, FSNT_idx] - pred[:, FSNS_idx] -
           pred[:, FLNT_idx] + pred[:, FLNS_idx])
    SHFLX = (inp[:, SHFLX_idx] * inp_div[SHFLX_idx] +
             inp_sub[SHFLX_idx])
    KEDINT = np.sum(dP_tilde * pred[:, DTVKE_idx], 1)

    # 4. Compute tendency of vapor due to phase change
    LHFLX = (inp[:, LHFLX_idx] * inp_div[LHFLX_idx] +
             inp_sub[LHFLX_idx])
    VAPINT = np.sum(dP_tilde * pred[:, PHQ_idx], 1)
    SPDQINT = (VAPINT - LHFLX) * L_S / L_V

    # 5. Same for cloud liquid water tendency
    SPDQCINT = np.sum(dP_tilde * pred[:, PHCLDLIQ_idx], 1) * L_I / L_V

    # 6. And the same for T but remember residual is still missing
    DTINT = np.sum(dP_tilde * pred[:, TPHYSTND_idx], 1)

    # 7. Compute enthalpy residual
    ENTRES = SPDQINT + SPDQCINT + DTINT - RAD - SHFLX - PHAS - KEDINT

    return np.square(ENTRES)

def lw_res_diagno(inp_div,inp_sub,norm_q,inp,pred):

    # Input
    PS_idx = 300

    # Output
    QRL_idx = slice(120, 150)
    FLNS_idx = 213
    FLNT_idx = 212

    # 1. Compute dP_tilde
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

    # 2. Compute longwave integral
    LWINT = np.sum(dP_tilde *pred[:, QRL_idx], axis=1)

    # 3. Compute net longwave flux from lw fluxes at top and bottom
    LWNET = pred[:, FLNS_idx] - pred[:, FLNT_idx]

    # 4. Compute water mass residual
    LWRES = LWINT-LWNET

    return np.square(LWRES)

def sw_res_diagno(inp_div,inp_sub,norm_q,inp,pred):

    # Input
    PS_idx = 300

    # Output
    QRS_idx = slice(150, 180)
    FSNS_idx = 211
    FSNT_idx = 210

    # 1. Compute dP_tilde
    dP_tilde = compute_dP_tilde(inp[:, PS_idx],  inp_div[PS_idx], inp_sub[PS_idx], norm_q, hyai, hybi)

    # 2. Compute longwave integral
    SWINT = np.sum(dP_tilde *pred[:, QRS_idx], axis=1)

    # 3. Compute net longwave flux from lw fluxes at top and bottom
    SWNET = pred[:, FSNT_idx] - pred[:, FSNS_idx]

    # 4. Compute water mass residual
    SWRES = SWINT-SWNET

    return np.square(SWRES)

def tot_res_diagno(inp_div,inp_sub,norm_q,inp,pred):
    return 0.25*(mass_res_diagno(inp_div,inp_sub,norm_q,inp,pred)+\
                ent_res_diagno(inp_div,inp_sub,norm_q,inp,pred)+\
                lw_res_diagno(inp_div,inp_sub,norm_q,inp,pred)+\
                sw_res_diagno(inp_div,inp_sub,norm_q,inp,pred))

mult_array = np.array([1,2,5,10,20])
end_array = ['_2','_2','','','']
dict_lay = {'SurRadLayer':SurRadLayer,'MassConsLayer':MassConsLayer,'EntConsLayer':EntConsLayer,
           'MassConsLayer_choice':MassConsLayer_choice,'EntConsLayer_choice':EntConsLayer_choice,
           'Conserving_Model':Conserving_Model,'weighted_loss':Weighted_Loss(lev1=14,lev2=104,multiplier=multiplier)}

NN = {}; md = {};
os.chdir(TRAINDIR+'/HDF5_DATA')
for im,multiplier in enumerate(mult_array):
    print('Multiplier is ',multiplier)
    path = TRAINDIR+'HDF5_DATA/Cm14_e14_multiplier'+str(multiplier)+end_array[im]+'.hdf5'
    NN[multiplier] = load_model(path,custom_objects=dict_lay)
    
gen = test_gen

SE = {}
TRES = {}
MSE = {}

spl = 0
while gen[spl][0].size>0: #spl is sample number
    
    print('spl=',spl,'                  ',end='\r')
    
    inp = gen[spl][0]
    truth = gen[spl][1]
    
    for im,multiplier in enumerate(mult_array):
        pred = NN[multiplier].predict_on_batch(inp)
        #pred[:,14] /= multiplier
        #pred[:,104] /= multiplier

        se = (pred-truth)**2

        pred_phys = pred/gen.output_transform.scale

        tresid = tot_res_diagno(gen.input_transform.div,gen.input_transform.sub,
                                gen.output_transform.scale[:30],inp,pred)

        if spl==0: SE[multiplier] = se; TRES[multiplier] = tresid; MSE[multiplier] = np.mean(se,axis=1);
        else: 
            SE[multiplier] += se; 
            TRES[multiplier] = np.concatenate((TRES[multiplier],tresid),axis=0); 
            MSE[multiplier] = np.concatenate((MSE[multiplier],np.mean(se,axis=1)),axis=0);

    spl += 1
    
for imultiplier,multiplier in enumerate(mult_array): SE[multiplier] /= spl
    
pathPKL = '/home/t/Tom.Beucler/SPCAM/CBRAIN-CAM/notebooks/tbeucler_devlog/PKL_DATA/'

hf = open(pathPKL+'2020_03_04_testgen036.pkl','wb')
S = {"TRES":TRES,"MSE":MSE,"SE":SE}
pickle.dump(S,hf)
hf.close()