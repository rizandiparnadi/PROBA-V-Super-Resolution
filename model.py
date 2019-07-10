import keras
from keras.layers import Input, UpSampling2D, Conv2D
from keras.models import Model
import keras.backend as K


def create_model(input_shape, activation='relu', num_filters=[128, 64], kernel_sizes=[9, 3, 5]):
    '''SRCNN architecture: https://arxiv.org/abs/1501.00092'''
    input = Input(shape=input_shape)

    x = Conv2D(num_filters[0], kernel_size=kernel_sizes[0], padding='same', activation=activation)(input)
    x = BatchNorm()(x)
    x = Conv2D(num_filters[1], kernel_size=kernel_sizes[1], padding='same', activation=activation)(x)
    x = BatchNorm()(x)
    x = Conv2D(1, kernel_size=kernel_sizes[2], padding='same', activation='linear')(x)

    model = Model(inputs=input, outputs=x)
    return model

def PSNR(y_true, y_pred):
    '''Peak Signal to Noise Ratio metric for Keras.'''
    epsilon = 1e-8
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true + epsilon), axis=-1)))) / 2.303

class BatchNorm(keras.layers.BatchNormalization):
    '''Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.
    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    '''
    def call(self, inputs, training=None):
        return super().call(inputs, training=False)