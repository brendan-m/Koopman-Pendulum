# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:27:38 2019

@original_author: dykua
Modified: brendan
Network architecture
"""

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Lambda, TimeDistributed, Conv1D, Flatten,Concatenate, Reshape, Add,BatchNormalization,Activation,Layer
import tensorflow as tf
from tensorflow.keras.regularizers import l1,l2 
from tensorflow.keras.initializers import Zeros,RandomUniform
import numpy as np
import keras.backend as K

def _transformer(x, out_dim, inter_dim_list=[32, 32],l2_reg=0, activation_out = 'linear'):
    for j in inter_dim_list:
        x = (Dense(j,
                   kernel_regularizer=l2(l2_reg),
                   bias_regularizer=l2(l2_reg),
                   activity_regularizer=l2(l2_reg),
                   bias_initializer=Zeros(),
                   kernel_initializer=RandomUniform(minval=-(1/np.sqrt(x.shape[-1])), maxval=(1/np.sqrt(x.shape[-1]))),

                  ))(x)
        x = (Activation('relu'))(x)
    x = (Dense(out_dim, 
               kernel_regularizer=l2(l2_reg),
               bias_regularizer=l2(l2_reg),
               activity_regularizer=l2(l2_reg),
               activation=activation_out,
               bias_initializer=Zeros(),
               kernel_initializer=RandomUniform(minval=-(1/np.sqrt(x.shape[-1])), maxval=(1/np.sqrt(x.shape[-1]))),

              ))(x)
    return x

# Undocumented in the paper, but original DKN code learns the eigenvalues from the magnitude of the complex latent coordinates
# I.e. for each complex pair,
# radius_of_pair = tf.reduce_sum(tf.square(pair_of_columns), axis=1, keep_dims=True)
# This class computes the magnitude of complex eigenfunctions, and concats with the real ones, as input to the aux network
class compute_aux_inputs(Layer):

    def __init__(self, num_complex, num_real, **kwargs):
        self.num_complex = num_complex
        self.num_real = num_real
        
        super(compute_aux_inputs, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_complex': self.num_complex,
            'num_real': self.num_real,
        })
        return config

    def build(self, input_shape):
        super(compute_aux_inputs, self).build(input_shape)

    def call(self, Gx):
        if self.num_complex:
            Gxc = Gx[:,:self.num_complex*2]
            Gxc = tf.reduce_sum(tf.reshape(tf.square(Gxc),(-1,self.num_complex,2)),axis=-1)
        if self.num_real:
            Gxr = Gx[:,2*self.num_complex:]
            
        if self.num_complex and self.num_real:
            return tf.concat([Gxc,Gxr], axis=-1)
        elif self.num_complex:
            return Gxc
        elif self.num_real:
            return Gxr
    
def _pred_K(x, num_complex, num_real,hidden_widths_omega, K_reg,l2_reg=0,activation_out='linear'):
    for j in hidden_widths_omega:
        x = (Dense(j,
                   kernel_regularizer=l2(l2_reg),
                   bias_regularizer=l2(l2_reg),
                   activity_regularizer=l2(l2_reg),
                   bias_initializer=Zeros(),
                   kernel_initializer=RandomUniform(minval=-(1/np.sqrt(x.shape[-1])), maxval=(1/np.sqrt(x.shape[-1]))),
                  ))(x) 
        x = (Activation('relu'))(x)
    Koop = (Dense(num_complex*2 + num_real,
                                 kernel_regularizer=l2(l2_reg),
                                 bias_regularizer=l2(l2_reg),
                                 activity_regularizer = l2(l2_reg),
                                 activation=activation_out,
                  kernel_initializer=RandomUniform(minval=-(1/np.sqrt(x.shape[-1])), maxval=(1/np.sqrt(x.shape[-1]))),
                  bias_initializer=Zeros(),
                 ))(x)
    return Koop

class JacobianLayer(Layer):
    def __init__(self, custom_Knet, **kwargs):
        self.custom_Knet = custom_Knet
        super(JacobianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(JacobianLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.custom_Knet(x)[1]  
        jacobian = tape.batch_jacobian(y, x)
        return jacobian

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])
    
class linear_update(Layer):

    def __init__(self, output_dim, num_complex, num_real,dt, **kwargs):
        self.output_dim = output_dim
        self.kernels = []
        self.num_complex = num_complex
        self.num_real = num_real
        self.dt = dt
        
        super(linear_update, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'kernels': self.kernels,
            'num_complex': self.num_complex,
            'num_real': self.num_real,
            'dt': self.dt,
        })
        return config

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(linear_update, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        y, Km = x # latent encoding and Koopman eigenvalues

        if self.num_complex:
            C_seq_c = [] 

            # for each complex pair
            for pair_index in range(0,2*self.num_complex,2):
                scale = tf.exp(Km[:,pair_index]*self.dt) # real component
                cs = tf.cos(Km[:,pair_index + 1]*self.dt) 
                sn = tf.sin(Km[:,pair_index + 1]*self.dt) 
                real = tf.multiply(scale, cs)
                img = tf.multiply(scale, sn)
                block = tf.stack([real, -img, img, real], axis = 1)
                Ci = tf.reshape(block, (-1, 2, 2))

                # Compute next timestep via multiplication with the block 
                y_c = y[:,pair_index:pair_index+2] # Part of latent encoding corresponding to current complex pair
                C_seq_c.append(tf.einsum('ik,ikj->ij', y_c, Ci)) 
  
            # Put all complex components into a single tensor
            C_seq_tensor = tf.reshape(tf.stack(C_seq_c, axis = 1), (-1, 2*self.num_complex))
       
        if self.num_real:
            R_seq = []
            R = tf.exp(Km[:,(2*self.num_complex):]*self.dt)
            R_seq_tensor = tf.multiply(y[:,(2*self.num_complex):], R)
        if self.num_complex and self.num_real:
            return tf.concat([C_seq_tensor, R_seq_tensor], axis=1)
        
        elif self.num_real:
            return R_seq_tensor
        elif self.num_complex:
            return C_seq_tensor
