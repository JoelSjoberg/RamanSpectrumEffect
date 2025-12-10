# For ORPL bubble fill algorithm, source is available at:
# Source: https://github.com/mr-sheg/orpl
# Guillaume Sheehy, Fabien Picot, Frédérick Dallaire, Katherine Ember, Tien Nguyen, Kevin Petrecca, Dominique Trudel, and Frédéric Leblond "Open-sourced Raman spectroscopy data processing package implementing a baseline removal algorithm validated from multiple datasets acquired in human tissue and biofluids," Journal of Biomedical Optics 28(2), 025002 (21 February 2023). https://doi.org/10.1117/1.JBO.28.2.025002

# Script containing methods usually used in my scripts, used to decrease cell size
import os, gc
import glob
import re
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import balanced_accuracy_score
np.set_printoptions(suppress=True, precision = 3)
import tensorflow as tf
from tensorflow.python.client import device_lib
print("Available computational components")
print(device_lib.list_local_devices())

# GPU support comes from tensorflow, so use builtin version of keras in tensorflow
# Change these depending on tf version
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Flatten, Dropout, Add, LSTM, GRU, Concatenate
from tensorflow.keras.layers import Conv1D, Input, Reshape, SpatialDropout1D, MaxPooling1D, LocallyConnected1D, ReLU
from tensorflow.keras.layers import Conv2D, SpatialDropout2D, MaxPooling2D, LocallyConnected2D, Lambda, GaussianNoise, Dot
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.activations import sigmoid, tanh, softmax, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, L1L2
import tensorflow.keras.initializers
import sklearn.metrics as metrics
#from BaselineRemovalCopy import *

from tensorflow.keras import regularizers
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def reset_seed(SEED = 0):
    
    """Reset the seed for every random library in use (System, numpy and tensorflow)"""
    
    os.environ['PYTHONHASHSEED']=str(SEED)
    np.random.seed(SEED)
    
    # Check tf version. some versions may have a different seed method!
    tf.random.set_seed(SEED)

def root_mean_squared_error(y_true, y_pred):
    
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):

    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# Edited R^2 function with minimum of 0
def r_square_loss(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (SS_res/(SS_tot + K.epsilon()) )

# Loss function of RMSE + R^2
def joined_loss(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred) + r_square_loss(y_true, y_pred)

def normalize(x):
    # Normalize list of spectra, of shape: (# samples, spectrum_length)
    min_ = np.expand_dims(np.min(x, axis=1), -1)
    max_ = np.expand_dims(np.max(x, axis=1), -1)
    
    return (x-min_)/(max_-min_ + 0.0000001)

def normalize_1D(x):
    # Normalize one spectrum
    # use nan to num to avoid division by 0 if spectrum is flat at 0
    return np.nan_to_num((x - np.min(x))/(np.max(x) - np.min(x)))

def extract_subsets(sample_subset, X, Y, LGM):
    # Take subset structure from the statistical tests
    #sample_subset = [ 0,  1,  8, 13, 14, 16, 20, 23, 26, 28, 30, 38, 44]
    # Sample subset is a list of integers containing the sample labels
    # X and Y are the data points and labels (one-hot encoded) respectively
    # LGM is the methylation classes we want, alternatively they are labels related to the research question
    new_sample_labels = {}
    
    new_x = np.empty((0, len(X[0])))
    new_y = np.empty((0, len(Y[0])))
    new_lgm = np.empty((0, len(LGM[0])))
    
    num_id = np.argmax(Y, axis = 1) # integer labels for locating subset indices
    
    for en, subset in enumerate(sample_subset):
        new_sample_labels[subset] = en

        ## Limit the data to the sample subset identified in the statistical tests

        new_x = np.concatenate([new_x, X[num_id == subset]])
        new_y = np.concatenate([new_y, Y[num_id == subset]])
        new_lgm = np.concatenate([new_lgm, LGM[num_id == subset]])

    # Form new one-hot encodings based on the number of labels
    subset_ohe = np.eye(len(sample_subset))
    new_y_ohe = []
    new_y = np.argmax(new_y, axis = 1)
    for n in range(len(new_y)):
        new_y_ohe.append(subset_ohe[new_sample_labels[new_y[n]]])
    new_y_ohe = np.array(new_y_ohe)
        
    # Shuffle the data afterwards
    
    np.random.seed(0)
    ix = np.arange(len(new_x))
    np.random.shuffle(ix)
    
    new_x = new_x[ix]
    new_y_ohe = new_y_ohe[ix]
    new_lgm = new_lgm[ix]

    # Return the sets
    return new_x, new_y_ohe, new_lgm

#Smallest allowable model we use when the samples are reduced significantly 
def make_model_small(lr = 0.0001, inp_size = 1738, out_dim = 13, reg_param = 1e-4, loss = "sparse_categorical_crossentropy"):
    
    reset_seed(SEED = 0)  

    inp = Input(shape = (inp_size,1))
    t = MaxPooling1D(79, 79)(inp)
    t = Flatten()(t)

    t = Dense(11,
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    t = Dense(out_dim, activation = "softmax",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)
    
    out = t

    model = Model(inp, out)
    model.compile(optimizer = Adam(learning_rate = lr), loss = loss, metrics = ["accuracy"])
    return model


# This is the architecture we use in predicting different classes, uniform except for the out_dim which depends on the target variable
def make_model(lr = 0.0001, inp_size = 1738, out_dim = 2, reg_param = 1e-5, loss = "sparse_categorical_crossentropy"):
    
    reset_seed(SEED = 0)  

    inp = Input(shape = (inp_size,1))
    t = Conv1D(6, 22, strides = 11, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(inp)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    t = Conv1D(7, 5, strides = 3, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    t = Conv1D(8, 3, strides = 3, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    t = MaxPooling1D(5)(t)
    t = Flatten()(t)
    t = tf.nn.l2_normalize(t, axis=1)
    
    t = Dense(out_dim,
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)

    t = tf.keras.activations.softmax(t)
    out = t

    model = Model(inp, out)
    model.compile(optimizer = Adam(learning_rate = lr), loss = loss, metrics = ["accuracy"])
    return model
    
def norm_initializer(shape, dtype=None):
    weights = tf.ones(shape, dtype = dtype)

    return weights

# After every update step, constrain each importance score to be <= 1, the feature importances are independent under this scaling
class ConstrainNormMax(tf.keras.constraints.Constraint):
    def __call__(self, weights):
        max_ = tf.reduce_max(tf.abs(weights))

        max_ = tf.clip_by_value(max_, 1, tf.float32.max)
        scaler = 1/max_

        return tf.abs(weights) * scaler


    def get_config(self):
        return {}

# Feature importance layer
class FeatureImportance1D(tf.keras.layers.Layer):  # Inherit from tf.keras.layers.Layer
    def __init__(self, input_dim=1738, increment = 1e-7, reg_param = 1e-3, **kwargs):
        super(FeatureImportance1D, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.reg_param = reg_param
        self.increment = increment
    def build(self, input_shape):
        
        self.importance = self.add_weight(
            shape=( self.input_dim, ),
            initializer = norm_initializer, 
            trainable = True,
            regularizer = regularizers.L1(self.reg_param),
            constraint = ConstrainNormMax(),
            
        )

        self.maximum_counter = self.add_weight(
            shape=(),
            initializer="zeros",
            trainable=False,
            aggregation=tf.VariableAggregation.NONE,
            name="maximum_counter"
        )
        self.maximum = self.add_weight(
            shape=( 1, ),
            initializer = norm_initializer, 
            trainable = False,
        name='maximum')

    # The function of this layer simply multiplies the learnable features with the input spectrum
    def call(self, inputs, training=None):
        
        activation = tf.math.multiply(inputs, self.importance)
        
        if training is True and self.maximum_counter < 1:
            self.maximum_counter.assign_add(self.increment)
            interpolation = self.maximum * self.maximum_counter + tf.reduce_max(activation) * (1-self.maximum_counter)
            self.maximum.assign(interpolation)
            #self.maximum.assign((self.maximum + tf.reduce_max(activation)) / 2)
            

        activation = activation / (self.maximum + 0.0000001)
        return activation

    def get_config(self):
        # Serialize the initializer and constraint
        config = super(FeatureImportance1D, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "reg_param": self.reg_param,
            "initializer": tf.keras.initializers.serialize(norm_initializer),
            "constraint": tf.keras.constraints.serialize(ConstrainNormMax()),
        })
        return config

# Negative categorical entropy loss for reducing ID-accuracy
def negative_CE(y_true, y_pred):
    return -tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred) * 0.1

"""
def make_split_model(lr = 0.00001,
                     inp_size = 1738,
                     out_dims = [13, 2],
                     reg_param = 1e-4,
                     losses = ["categorical_crossentropy", "categorical_crossentropy"]):
    
    reset_seed(SEED = 0)  
    inp = Input(shape = (inp_size,1))
    t = Conv1D(2, 22, strides = 22, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(inp)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    t = Conv1D(4, 60, strides = 2, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    t = Conv1D(8, 9, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)
    t = BatchNormalization()(t)
    t_split = LeakyReLU()(t)

    t_split = Flatten()(t_split)
    t_id = Dense(out_dims[0],
        kernel_regularizer=regularizers.L2(reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param), activation = "softmax", name = "ID")(t_split)

    t_lgm = Dense(out_dims[1],
        kernel_regularizer=regularizers.L2(reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param), activation = "softmax", name = "LGm")(t_split)
    

    out = [t_id, t_lgm]

    # Store output in a list which we return
    output = out
    
    model = Model(inp, output)
    model.compile(
        optimizer= Adam(learning_rate=lr),
        loss = {"ID" : losses[0], "LGm" : losses[1]},
        metrics = ["accuracy"],
    )
    return model
"""

def make_split_model(lr = 0.00001,
                     inp_size = 1738,
                     out_dims = [46, 2],
                     reg_param = 1e-5,
                     losses = ["categorical_crossentropy", "categorical_crossentropy"]):
    
    reset_seed(SEED = 0)  
    inp = Input(shape = (inp_size,1))
    t = Conv1D(6, 22, strides = 11, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(inp)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    t = Conv1D(7, 5, strides = 3, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    t = Conv1D(8, 3, strides = 3, padding = "valid",
        kernel_regularizer=regularizers.L2(l2=reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    t = MaxPooling1D(5)(t)

    t_split = Flatten()(t)
    t_split = tf.nn.l2_normalize(t_split, axis=1)
    
    t_id = Dense(out_dims[0],
        kernel_regularizer=regularizers.L2(reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param), activation = "softmax", name = "ID")(t_split)

    t_lgm = Dense(out_dims[1],
        kernel_regularizer=regularizers.L2(reg_param),
        bias_regularizer=regularizers.L2(reg_param),
        activity_regularizer=regularizers.L2(reg_param), activation = "softmax", name = "LGm")(t_split)
    

    out = [t_id, t_lgm]

    # Store output in a list which we return
    output = out
    
    model = Model(inp, output)
    model.compile(
        optimizer= Adam(learning_rate=lr),
        loss = {"ID" : losses[0], "LGm" : losses[1]},
        metrics = ["accuracy"],
    )
    return model

def make_combined_model(enc,
                     sample_model, # This model should have 2 outputs
                     inp_shape = 1738,
                     lr = 0.0001,
                     losses = ["categorical_crossentropy", "categorical_crossentropy"],
                    metrics = ["accuracy"]):

    reset_seed(SEED = 0)  
    
    inp = Input(shape = (inp_shape,1))

    t = enc(inp)
    
    t = tf.expand_dims(t, -1)
 
    id, lgm = sample_model(t)
    id._name = "ID"
    lgm._name = "LGm"
    
    out = [id, lgm]
    
    model = Model(inp, out)
    
    model.compile(
        optimizer= Adam(learning_rate=lr),
        loss=[losses[0], losses[1]],       # list, not dict
        metrics=["accuracy"]
        #loss = {"ID" : losses[0], "LGm" : losses[1]},
        #metrics={"ID": "accuracy", "LGm": "accuracy"}
    )
    return model

# The encoder is used to transform the data using only the learned features, normalization is optional
def make_encoder(lr = 0.0001, inp_shape = 1738, feature_max_increment = 1e-7):
    
    reset_seed(SEED = 0)

    inp = Input(shape = (inp_shape,1))
    
    t = Flatten()(inp)

    t = FeatureImportance1D(inp_shape, feature_max_increment, name = "importance")(t)
    t = tf.expand_dims(t, axis = -1)
    t = AveragePooling1D(pool_size = 25, strides = 1, padding = "same")(t)
    t = tf.nn.l2_normalize(t, axis=1)
    output = t
    
    model = Model(inp, output)
    
    model.compile(
        optimizer= Adam(learning_rate=lr),
        loss = root_mean_squared_error,
        metrics = ["accuracy"]
    )
    return model


class CustomPad(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super(CustomPad, self).__init__(**kwargs)
        self.mid = int(size / 2)
        self.remainder = size - self.mid * 2 # If size is uneven

    def call(self, inputs, training=False):
        shape = tf.shape(inputs)
        seq_length = shape[1]
        
        left_column = tf.gather(inputs, [0], axis=1)
        right_column = tf.gather(inputs, [seq_length - 1], axis=1)
        
        padding_left = tf.tile(left_column, [1, self.mid, 1])
        padding_right = tf.tile(right_column, [1, self.mid -1 + self.remainder, 1])
        
        out = tf.concat([padding_left, inputs, padding_right], axis=1)
        
        return out
    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.mid,
        })
        return config



def make_ensemble(lr=0.00001):
    
    reset_seed(SEED = 0)  
    scaler = 10
    dim_red_size = 12**2
    l1_param = 1e-6
    l2_param = 1e-6
    
    inp = Input(shape = (None,1))
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(inp)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    inp_key = Conv1D(filters = dim_red_size, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    inp_key = GlobalMaxPooling1D()(inp_key)
    inp_key = tf.keras.backend.expand_dims(inp_key, -1)
    
    t_dot = Dot(axes=(2, 2))([inp, inp_key])
    t = tf.keras.backend.expand_dims(t_dot, -1)
    
    kernel_size = 32
    t = Conv2D(filters = 1 * scaler,
               kernel_size = (kernel_size, int(dim_red_size/10)),
               strides = (1, int(np.sqrt(dim_red_size))),
               padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 16
    t_res = Conv2D(filters = 1 * scaler, 
              kernel_size = (kernel_size, int(dim_red_size/10)),
              strides = (1, int(np.sqrt(dim_red_size))),
              padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t_res)
    t = LeakyReLU()(t)
    
    t = Reshape((-1, scaler))(t)
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)

    # Baseline
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    b = Conv1D(filters = 1, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    
    pool_size = 33
    padded = CustomPad(pool_size)(b)
    b = AveragePooling1D(pool_size, 1, padding = "valid")(padded)
    b = tf.keras.activations.relu(b)
    
    # Cosmic rays
    cr_size = 3
    cr_padded = CustomPad(cr_size)(t)
    cr = Conv1D(filters = 1, kernel_size = cr_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(cr_padded)

    cr = tf.keras.activations.relu(cr)

    ### Second part: Extract peaks and noise from the input
    reduced_spectrum = Add()([inp, -b, -cr])
    kernel_size = 128
    padded = CustomPad(kernel_size)(reduced_spectrum)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    inp_key = Conv1D(filters = dim_red_size, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    inp_key = GlobalMaxPooling1D()(inp_key)
    inp_key = tf.keras.backend.expand_dims(inp_key, -1)
    
    t_dot = Dot(axes=(2, 2))([reduced_spectrum, inp_key])
    t = tf.keras.backend.expand_dims(t_dot, -1)
    
    kernel_size = 32
    t = Conv2D(filters = 1 * scaler,
               kernel_size = (kernel_size, int(dim_red_size/10)),
               strides = (1, int(np.sqrt(dim_red_size))),
               padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 16
    t = Conv2D(filters = 1 * scaler, 
              kernel_size = (kernel_size, int(dim_red_size/10)),
              strides = (1, int(np.sqrt(dim_red_size))),
              padding = "same",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)
    
    t = Add()([t, t_res])
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    t = Reshape((-1, scaler))(t)
    
    kernel_size = 128
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    kernel_size = 64
    padded = CustomPad(kernel_size)(t)
    t = Conv1D(filters = 1 * scaler, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=l1_param, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)
    t = BatchNormalization()(t)
    t = LeakyReLU()(t)
    
    
    # Peaks, make the kernel size small to deal with potentially sharp peaks
    kernel_size = 8 
    padded = CustomPad(kernel_size)(t)
    p = Conv1D(filters = 1, kernel_size = kernel_size, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(padded)

    pool_size = 4
    padded = CustomPad(pool_size)(p)
    p = AveragePooling1D(pool_size, 1, padding = "valid")(padded)
    p = tf.keras.activations.relu(p)

    n = Conv1D(filters = 1, kernel_size = 1, strides = 1, padding = "valid",
              kernel_regularizer= L1L2(l1=1e-5, l2=l2_param),
              bias_regularizer= l2(l2_param),
              activity_regularizer= l2(l2_param))(t)

    n = tf.keras.activations.tanh(n)


    # Flatten each part to make them comparable to the labels
    b = Flatten()(b)
    cr = Flatten()(cr)
    n = Flatten()(n)
    p = Flatten()(p)
    
    # Store output in a list which we return
    output = [b, cr, n, p]
    
    model = Model(inp, output)
    
    model.compile(
        optimizer= Adam(learning_rate=lr),
        loss= joined_loss,
        metrics = [
            root_mean_squared_error,
            r_square,
            ])

    return model


