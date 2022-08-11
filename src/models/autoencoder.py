import numpy as np
import matplotlib.pyplot as plt
import time as time
import tensorflow as tf
import tensorflow.keras.models as Models
import tensorflow.keras.layers as Layers
from src.models.MDN_models import MDN
import tensorflow.keras.activations as Activations
import tensorflow.keras.backend as K
from tensorflow_probability import distributions as tfd


class Normalizedcomplex(Layers.Layer):
    def __init__(self):
        super(Normalizedcomplex, self).__init__(name='Normalizedcomplex')
    def build(self, input_shape):
        self.N = int(input_shape[1]/2)
    def call(self, inputs):
        x=inputs
        x = tf.nn.l2_normalize(inputs, axis=1)*np.sqrt(2)
        #x= x / K.sqrt(K.mean(x**2))
        return x
    

class Encoder(tf.keras.Model):
    def __init__(self, M, ns):
        super(Encoder, self).__init__(name='Encoder')
        self.dense1 = Layers.Dense(M, activation='relu')
        self.dense2 = Layers.Dense(M, activation='relu')
        self.dense3 = Layers.Dense(2*ns, activation='linear')
        self.norm = Layers.Lambda(lambda x: Activations.tanh(x / K.sqrt(K.mean(x**2)))) # new act that allows 16QAM comparison
        
    def call(self, inputs):
        return self.norm(self.dense3(self.dense2(self.dense1(inputs))))

class MDN_model(tf.keras.Model):
    def __init__(self, N_HIDDEN, OUTPUT_DIMS, N_MIXES):
        super(MDN_model, self).__init__(name='MDN_Channel')
        self.dense1 = Layers.Dense(N_HIDDEN,activation='relu')
        self.dense2 = Layers.Dense(N_HIDDEN,activation='relu')
        self.mdn=MDN(OUTPUT_DIMS, N_MIXES)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        mdn_out = self.mdn(x)
        return mdn_out

    
def sampling_mdn(y_pred,num_mixes,output_dim):
    y_pred = tf.reshape(y_pred, [-1, (2 * num_mixes * output_dim) + num_mixes], name='reshape_ypreds')
    out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mixes * output_dim,
                                                                         num_mixes * output_dim,
                                                                         num_mixes],
                                             axis=1, name='mdn_coef_split')
    cat = tfd.Categorical(logits=out_pi)
    component_splits = [output_dim] * num_mixes
    mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
    sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
    coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
    mixture = tfd.Mixture(cat=cat, components=coll)
    samp = mixture.sample()
    return samp

# Nnet_decoder_input = Layers.Input(shape=[number_symbols+2*number_silence], dtype='complex64')
class Decoder(tf.keras.Model):
    def __init__(self, nb,M):
        super(Decoder, self).__init__(name='Decoder')
        self.dense2 = Layers.Dense(M, activation='relu')
        self.dense3 = Layers.Dense(M, activation='softmax')
    def call(self, inputs):
        x = self.dense2(inputs)
        x = self.dense3(x)
        return x
    

class Autoencoder(tf.keras.Model):
    def __init__(self, MDN, nb, ns, nsi,N_HIDDEN, OUTPUT_DIMS, N_MIXES):
        super(Autoencoder, self).__init__(name='Autoencoder')
        self.encoder = Encoder(M, ns)
       # self.channel = Layers.GaussianNoise(0.2)
        self.channel = MDN
        self.sampling = Layers.Lambda(lambda x: sampling_mdn(x,N_MIXES,OUTPUT_DIMS))
        self.decoder = Decoder(nb,M)
    def call(self, inputs):
        Symbols = inputs
        x1 = self.encoder(Symbols)
        x = self.channel(x1)
        x= self.sampling(x)
        x = self.decoder(x)
        return x
    
    def sample(self,z):
        return self.encoder(z)
    
    def decoder_predict(self,z):
        return self.decoder(z)
    
    def trainable_check(self):
        return self.channel.trainable
    
class CapacityLoss(tf.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return -tf.math.reduce_mean(tf.math.log((1+y_pred*y_true)/2), axis=1)
    