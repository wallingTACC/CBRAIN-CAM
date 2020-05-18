"""
Define all different types of models.

<<<<<<< HEAD
Author: Stephan Rasp
tgb - 2/7/2019 - Adding mass and enthalpy conservation layers as models
tgb - 2/7/2019 - Replacing keras with tf.keras to avoid incompatibilities when using tensorflow's eager execution
"""

from cbrain.imports import *
#import keras
import tensorflow as tf
from  tensorflow import math as tfm
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from cbrain.losses import *
from cbrain.cam_constants import *
from cbrain.layers import *
act_dict = tensorflow.keras.activations.__dict__
lyr_dict = tensorflow.keras.layers.__dict__

L_V = 2.501e6   # Latent heat of vaporization
L_I = 3.337e5   # Latent heat of freezing
L_S = L_V + L_I # Sublimation
C_P = 1.00464e3 # Specific heat capacity of air at constant pressure
G = 9.80616
P0 = 1e5

def cons_5dens():
# tgb - 2/7/2019 - Draft of the energy/mass conserving model
# Improve it using fc_model
    inp = Input(shape=(304,))
    densout = Dense(512, activation='linear')(inp)
    densout = LeakyReLU(alpha=0.3)(densout)
    for i in range (4):
        densout = Dense(512, activation='linear')(densout)
        densout = LeakyReLU(alpha=0.3)(densout)
    densout = Dense(156, activation='linear')(densout)
    densout = LeakyReLU(alpha=0.3)(densout)
    massout = MasConsLay(
        input_shape=(156,), fsub=fsub, fdiv=fdiv, normq=normq,\
        hyai=hyai, hybi=hybi, output_dim = 157
    )([inp, densout])
    out = EntConsLay(
        input_shape=(157,), fsub=fsub, fdiv=fdiv, normq=normq,\
        hyai=hyai, hybi=hybi, output_dim = 158
    )([inp, massout])
    
    return tf.keras.models.Model(inp, out)    
    
class PartialReLU(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a = x[:, :60]
        b = K.maximum(x[:, 60:120], 0)
        c = x[:, 120:]
        return K.concatenate([a, b, c], 1)

    def compute_output_shape(self, input_shape):
        return input_shape

class QLayer(Layer):
    def __init__(self, fsub, fdiv, hyai, hybi, **kwargs):
        super().__init__(**kwargs)
        self.fsub, self.fdiv, self.hyai, self.hybi = fsub, fdiv, np.array(hyai), np.array(hybi)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, arrs):
        L_V = 2.501e6 ; L_I = 3.337e5; L_S = L_V + L_I
        C_P = 1.00464e3; G = 9.80616; P0 = 1e5

        f, a = arrs
        # Get pressure difference 
        PS = f[:, 90] * self.fdiv[90] + self.fsub[90]
        P = P0 * self.hyai + PS[:, None] * self.hybi
        dP = P[:, 1:] - P[:, :-1]

        # Get Convective integrals
        iPHQ = a[:, 30:60]*dP/G/L_S
        vPHQ = K.sum(iPHQ, 1)
        absvPHQ = K.sum(K.abs(iPHQ),1)
        # Sum for convective moisture
        dQCONV = vPHQ

        # Get surface flux
        LHFLX = (f[:, 93] * self.fdiv[93] + self.fsub[93]) / L_V

        # Get precipitation sink
        TOT_PRECL = a[:, 64] / (24*3600*2e-2)

        # Total differences to be corrected --> factor. Correct everything involved
        #pdb.set_trace()
        dQ = dQCONV - LHFLX + TOT_PRECL
        absTOT = absvPHQ + K.abs(TOT_PRECL)
        # Correct PHQ
        fact = dQ[:, None] * K.abs(iPHQ) / absTOT[:, None]
        b = a[:, 30:60] - fact / dP*G*L_S
        # Correct precipitation sink
        fact = dQ[:] * K.abs(TOT_PRECL) / absTOT[:]
        c = a[:, 64] - fact * (24*3600*2e-2)

        return K.concatenate([a[:, :30], b, a[:, 60:64], c[:, None]], 1)

    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv), 
                  'hyai': list(self.hyai), 'hybi': list(self.hybi)}
        base_config = super(QLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[1]

class ELayer(Layer):
    def __init__(self, fsub, fdiv, hyai, hybi, **kwargs):
        super().__init__(**kwargs)
        self.fsub, self.fdiv, self.hyai, self.hybi = fsub, fdiv, np.array(hyai), np.array(hybi)

    def build(self, input_shape):
        super().build(input_shape)  # Be sure to call this somewhere!

    def call(self, arrs):
        L_V = 2.501e6 ; L_I = 3.337e5; L_S = L_V + L_I
        C_P = 1.00464e3; G = 9.80616; P0 = 1e5

        f, a = arrs
        # Get pressure difference 
        PS = f[:, 90] * self.fdiv[90] + self.fsub[90]
        P = P0 * self.hyai + PS[:, None] * self.hybi
        dP = P[:, 1:] - P[:, :-1]

        # Get Convective integrals
        iTPHY, iPHQ = a[:, :30]*dP/G, a[:, 30:60]*dP/G/L_S*L_V
        vTPHY, vPHQ = K.sum(iTPHY, 1), K.sum(iPHQ, 1)
        absvTPHY, absvPHQ = K.sum(K.abs(iTPHY),1), K.sum(K.abs(iPHQ),1)

        # Get surface fluxes
        SHFLX = f[:, 92] * self.fdiv[92] + self.fsub[92]
        LHFLX = (f[:, 93] * self.fdiv[93] + self.fsub[93])

        # Radiative fluxes
        dERADFLX = K.sum(a[:, -5:-1], 1) * 1e3
        absRADFLX = K.sum(K.abs(a[:, -5:-1]), 1) * 1e3

        # Total differences to be corrected --> factor. Correct heating and 2D terms
        dE = vTPHY - SHFLX - dERADFLX + vPHQ - LHFLX
        absTOT = absvTPHY + absRADFLX
        # Correct TPHY
        fact = dE[:, None] * K.abs(iTPHY) / absTOT[:, None]
        b = a[:, :30] - fact / dP*G
        # Correct Radiative fluxes
        fact = dE[:, None] * K.abs(a[:, -5:-1]) * 1e3 / absTOT[:, None]
        c = a[:, -5:-1] + fact / 1e3

        return K.concatenate([b, a[:, 30:60], c, a[:, -1][:, None]], 1)

    def get_config(self):
        config = {'fsub': list(self.fsub), 'fdiv': list(self.fdiv), 
                  'hyai': list(self.hyai), 'hybi': list(self.hybi)}
        base_config = super(ELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[1]

def act_layer(act):
    """Helper function to return regular and advanced activation layers"""
    act = Activation(act) if act in tf.keras.activations.__dict__.keys() \
        else tf.keras.layers.__dict__[act]()
    return act


def fc_model(input_shape, output_shape, hidden_layers, activation, conservation_layer=False,
             inp_sub=None, inp_div=None, norm_q=None):
    inp = Input(shape=(input_shape,))

    # First hidden layer
    x = Dense(hidden_layers[0])(inp)
    x = act_layer(activation)(x)

    # Remaining hidden layers
    for h in hidden_layers[1:]:
        x = Dense(h)(x)
        x = act_layer(activation)(x)

    if conservation_layer:
        x = SurRadLayer(inp_sub, inp_div, norm_q)([inp, x])
        x = MassConsLayer(inp_sub, inp_div, norm_q)([inp, x])
        out = EntConsLayer(inp_sub, inp_div, norm_q)([inp, x])

    else:
        out = Dense(output_shape)(x)

    return tf.keras.models.Model(inp, out)


