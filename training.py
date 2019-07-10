import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import functools
import shutil
import skimage
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K

from keras_superconvergence import LRFinder, CyclicalLR, cycles
from keras_superconvergence.utils import PerBatchMetrics, PerEpochMetrics
from dataset import ProbaVDataset
from model import BatchNorm, PSNR


def lr_finder(model, dataset, hyperparams, custom_objects={}):
    '''Finds optimal learning rate for cyclical learning rates with LRFinder.
    Args:
        model: Keras model
        dataset: Dataset class implementing generate() that outputs batches of tensors
        hyperparams (dict): Hyperparameters in a dictionary
    '''
    model.compile(optimizer=hyperparams['optimizer'], loss=hyperparams['loss_fn'])
    lr_finder = LRFinder(model, custom_objects=custom_objects)
    lr_finder.find_generator(dataset.generate(augment=False), start_lr=hyperparams['start_lr'], end_lr=hyperparams['end_lr'], 
                            max_loss=hyperparams['max_loss'], steps_per_epoch=dataset.steps_per_epoch)
    plt.figure(figsize=(10,5))
    lr_finder.plot_losses()
    plt.show()
    plt.savefig('lr_finder.png')

def train(mode, model, dataset, hyperparams):
    '''
    Args:
        mode (str): Is either:
            `standard`: train with normal learning rate and decay.
            `clr`: train with cyclical learning rates.
            `debug`: overfits on a single batch for debugging.
        model: Keras model
        dataset: Dataset class implementing generate() that outputs batches of tensors
        hyperparams (dict): Contains hyperparameters
    Returns: Trained model
    '''
    batch_metrics = PerBatchMetrics()
    epoch_metrics = PerEpochMetrics()
    dataset.reset_for_training()

    if mode=='clr':
        cycle = functools.partial(cycles.linear_then_cosine, slant=0.75)
        num_epochs = hyperparams['num_epochs']
        clr = CyclicalLR(cycle_fn=cycle, max_lr=hyperparams['max_lr'], 
                        max_momentum=hyperparams['max_momentum'], div_factor=hyperparams['div_factor'],
                        num_cycles=hyperparams['num_cycles'], num_epochs=num_epochs, 
                        steps_per_epoch=dataset.steps_per_epoch)

        # Train
        model.compile(optimizer=hyperparams['optimizer'], loss=hyperparams['loss_fn'], metrics=[PSNR])
        model.fit_generator(dataset.generate(), epochs=num_epochs, validation_data=dataset.validation_data,
                            steps_per_epoch=dataset.steps_per_epoch, callbacks=[clr, batch_metrics, epoch_metrics], verbose=True)
        plot_metrics(batch_metrics, epoch_metrics)
        return model

    elif mode=='standard':
        plateau = ReduceLROnPlateau(patience=5)
        
        model.compile(optimizer=hyperparams['optimizer'], loss=hyperparams['loss_fn'], metrics=[PSNR])
        model.fit_generator(dataset.generate(), epochs=hyperparams['num_epochs'], validation_data=dataset.validation_data,
                            steps_per_epoch=dataset.steps_per_epoch, callbacks=[plateau, batch_metrics, epoch_metrics], verbose=True)
        plot_metrics(batch_metrics, epoch_metrics)
        return model

    elif mode=='debug':
        '''Quick sanity check: overfit on a single batch as '''
        val_data = next(dataset.generate(augment=False, move_on=False))

        cycle = functools.partial(cycles.linear_then_cosine, slant=0.75)
        num_epochs = hyperparams['num_epochs']
        clr = CyclicalLR(cycle_fn=cycle, max_lr=hyperparams['max_lr'], 
                        max_momentum=hyperparams['max_momentum'], div_factor=hyperparams['div_factor'],
                        num_cycles=hyperparams['num_cycles'], num_epochs=1, 
                        steps_per_epoch=hyperparams['num_epochs'])

        model.compile(optimizer=hyperparams['optimizer'], loss=hyperparams['loss_fn'], metrics=[PSNR])
        model.fit_generator(dataset.generate(augment=False, move_on=False), epochs=1, validation_data=val_data,
                            steps_per_epoch=hyperparams['epochs'], callbacks=[batch_metrics, epoch_metrics], verbose=True)
        plot_metrics(batch_metrics, epoch_metrics)
        return model


def plot_metrics(batch_metrics, epoch_metrics, start_iter=0):
    '''Plots training metrics.'''
    has_validation = 'val_loss' in epoch_metrics.logs

    plt.figure(figsize=(15, 5))
    plt.title('Loss')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.xlabel('Iterations')
    plt.ylabel('loss')
    plt.plot(batch_metrics.iteration[start_iter:], batch_metrics.logs['loss'][start_iter:])
    if has_validation: plt.plot(epoch_metrics.iteration, epoch_metrics.logs['val_loss'])

    plt.figure(figsize=(15, 5))
    plt.title('PSNR')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.plot(batch_metrics.iteration[start_iter:], batch_metrics.logs['PSNR'][start_iter:])
    if has_validation: plt.plot(epoch_metrics.iteration, epoch_metrics.logs['val_PSNR'])
    
    plt.figure(figsize=(15, 5))
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.title("Learning Rate")
    plt.plot(batch_metrics.iteration[start_iter:], batch_metrics.lr[start_iter:])


def predict(model, data, batch_size=8, display_n=None):
    '''Do inference on data.
    Args:
        model: Keras model
        data: (X, Y) data tuple. Y can be None
        display_n: How many images to display. If none, display all
    Returns: Prediction tensor Y
    '''
    X, Y = data
    Y_pred = model.predict(X, batch_size=batch_size, verbose=True)

    if display_n==None: display_n = Y_pred.shape[0]
    if Y is None: cols = 2
    else: cols = 3
    for i in range(display_n):
        lr = X[i]
        sr = Y_pred[i].squeeze()
        vmin, vmax = lr.min(), lr.max()
        plt.figure(figsize=(20,7))
        plt.subplot(1, cols, 1)
        plt.imshow(lr[:,:,0], vmin=vmin, vmax=vmax)
        plt.title('LR')
        plt.subplot(1, cols, 2)
        plt.imshow(sr, vmin=vmin, vmax=vmax)
        plt.title('SR')

        if Y is not None:
            hr = Y[i].squeeze()
            plt.subplot(1, cols, 3)
            plt.imshow(hr, vmin=vmin, vmax=vmax)
            plt.title('HR')

        plt.show()

    return Y_pred

def predict_on_test(model, dataset, path='submission', display_n=None):
    '''Do inference on test set, display and save to directory for submission.'''
    if not os.path.exists(path):
        os.makedirs(path)
    
    i = 0
    for _ in range(dataset.test_steps):
        X, _,  = next(dataset.generate(which='test', augment=False))
        Y_pred = model.predict_on_batch(X)
        for scene in range(X.shape[0]):
            x = X[scene].squeeze()
            y = Y_pred[scene].squeeze()
            # Show
            if i < display_n:
                vmin, vmax = x.min(), x.max()
                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(x[:,:,0], vmin=vmin, vmax=vmax)
                plt.title('Low Resolution')
                plt.subplot(1, 2, 2)
                plt.imshow(y, vmin=vmin, vmax=vmax)
                plt.title('Super-Resolved')
                plt.show()
            # Save to file
            scene_id = dataset.test_scenes[i].id
            filename = path + '/imgset{:04d}.png'.format(scene_id)
            cv2.imwrite(filename, skimage.img_as_uint(y))
            i += 1

    # Zip it
    shutil.make_archive('submission', 'zip', path)