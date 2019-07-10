import keras
from keras.layers import Input, UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Concatenate, Dropout, BatchNormalization, Activation
from keras.models import Model
import keras.backend as K


def create_model(input_shape, activation='relu', long_skip='Add'):
    '''Custom architecture. See project report for details.'''
    input = Input(shape=input_shape)

    # Head block
    x = Conv2D(64, (3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(input)
    x = BatchNorm()(x)
    x = Conv2D(64, (3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(x)
    x = BatchNorm()(x)
    skip = x

    # Mini U-Net block
    encoder_0, encoder_0_pool = encoder_block(x, 64, activation=activation) # 128^2
    encoder_1, encoder_1_pool = encoder_block(encoder_0_pool, 64, activation=activation) # 64^2
    encoder_2, encoder_2_pool = encoder_block(encoder_1_pool, 128, activation=activation) # 32^2
    encoder_3, encoder_3_pool = encoder_block(encoder_2_pool, 256, activation=activation) # 16^2

    center = center_block(encoder_3_pool, 512, activation=activation) # 8^2

    decoder_0 = decoder_block(center, encoder_3, 256, activation=activation) # 16^2
    decoder_1 = decoder_block(decoder_0, encoder_2, 128, activation=activation) # 32^2
    decoder_2 = decoder_block(decoder_1, encoder_1, 64, activation=activation) # 64^2
    decoder_3 = decoder_block(decoder_2, encoder_0, 64, activation=activation, end_with_activation=False) # 128^2
    
    if long_skip=='Add':
      x = Add()([decoder_3, skip])
      x = Activation(activation)(x)
    elif long_skip=='Concatenate':
      x = Activation(activation)(decoder_3)
      x = Concatenate()([x, skip])
    else:
      raise ValueError('Invalid long_skip:', long_skip)

    # Upsample block
    x = Conv2DTranspose(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(x)
    x = BatchNorm()(x)
    x = Conv2D(64, (3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(x)
    x = BatchNorm()(x)
    x = Conv2D(64, (3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(x)
    x = BatchNorm()(x)
    x = Conv2D(1, (1, 1), padding='same', activation='linear', kernel_initializer='he_normal')(x)

    model = Model(inputs=input, outputs=x)
    return model

def residual_block(input_tensor, num_filters, activation='relu', end_with_activation=True):
    x = Conv2D(num_filters, (3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(input_tensor)
    x = BatchNorm()(x)
    x = Conv2D(num_filters, (3, 3), padding='same', activation='linear', kernel_initializer='he_normal')(x)
    x = BatchNorm()(x)
    x = shortcut(input_tensor, x, activation=activation)
    if end_with_activation:
      x = Activation(activation)(x)
    return x

def shortcut(shortcut_tensor, residual_tensor, activation='relu'):
    '''Residual block's skip connection .If filters are the same size, we can simply add, 
    but if they are different, we do 1x1 convolution first to match tensor shapes.'''
    shortcut_shape = K.int_shape(shortcut_tensor)
    residual_shape = K.int_shape(residual_tensor)
    if shortcut_shape!=residual_shape:
        num_filters = residual_shape[-1]
        shortcut_tensor = Conv2D(num_filters, (1, 1), padding='same', activation=activation, kernel_initializer='he_normal')(shortcut_tensor)
    return Add()([shortcut_tensor, residual_tensor])

def encoder_block(input_tensor, num_filters, activation='relu', downsample_type='MaxPooling2D'):
    encoder = residual_block(input_tensor, num_filters)
    if downsample_type=='MaxPooling2D':
        encoder_pooled = MaxPooling2D(pool_size=(2, 2))(encoder)
    elif downsample_type=='Conv2D':
        encoder_pooled = Conv2D(num_filters, kernel_size=(2, 2), strides=(2, 2), padding='valid', activation=activation, kernel_initializer='he_normal')(encoder)
    else:
        raise ValueError('Unknown downsample_type:', downsample_type)
    return encoder, encoder_pooled

def center_block(input_tensor, num_filters, activation='relu'):
    block = residual_block(input_tensor, num_filters, activation=activation)
    return block

def decoder_block(input_tensor, skip_tensor, num_filters, activation='relu', upsample_type='Conv2DTranspose', skip_type='Concatenate', end_with_activation=True):
    if upsample_type=='UpSampling2D':
        decoder = UpSampling2D(size=(2, 2))(input_tensor)
    elif upsample_type=='Conv2DTranspose':
        decoder = Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same', activation=activation, kernel_initializer='he_normal')(input_tensor)
    else:
        raise ValueError('Unknown upsample_type:', upsample_type)
        
    if skip_type=='Concatenate':
        decoder = Concatenate(axis=-1)([decoder, skip_tensor])
    elif skip_type=='Add':
        decoder = Add()([decoder, skip_tensor])
    else:
        raise ValueError('Unknown skip_type:', skip_type)
        
    decoder = residual_block(decoder, num_filters, activation=activation, end_with_activation=end_with_activation)
    return decoder

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