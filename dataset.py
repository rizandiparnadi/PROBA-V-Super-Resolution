import numpy as np 
from PIL import Image
import cv2
import math
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import skimage
from skimage.feature import register_translation
import probav.embiggen
import warnings


class ProbaVDataset():
    def __init__(self, data_path='probav_data', batch_size=8, validation_split=0.1, upsample_input=False):
        '''Dataset class. Loads data and passes it in batches for training/testing.
        Args:
            data_path (str): Path to data
            batch_size (int): Batch size for training
            validation_split (float): Fraction of training data split for validation set
            upsample_input (bool): Controls how we generate input data with generate() for training:
                If False, it returns n plain low resolution images stacked as n channels.
                If True, it returns one channel per-pixel median of n images, upsampled bicubicly.
        '''
        self.train_path = data_path + '/train/'
        self.test_path = data_path + '/test/'
        self.train_red_range = (0, 594)
        self.train_nir_range = (594, 1160)
        self.test_red_range = (1160, 1306)
        self.test_nir_range = (1306, 1450)
        self.upsample_input = upsample_input

        train_val_scenes = self.load_scenes(self.train_path, self.train_red_range, self.train_nir_range)
        split_point = int((1.0-validation_split)*len(train_val_scenes))
        self.train_scenes = train_val_scenes[:split_point]
        self.val_scenes = train_val_scenes[split_point:]
        self.test_scenes = self.load_scenes(self.test_path, self.test_red_range, self.test_nir_range, is_test=True)

        val_X, val_Y, val_sms, val_ids = self.get_batch(self.val_scenes)
        self.validation_data = val_X, val_Y
        self.validation_sms = val_sms
        self.validation_scene_ids = val_ids
        if len(self.val_scenes)==0:
            self.validation_data = None

        self.batch_size = batch_size
        self.steps_per_epoch = math.ceil(float(len(self.train_scenes))/float(self.batch_size))
        self.test_steps = math.ceil(float(len(self.test_scenes))/float(self.batch_size))
        self.train_step = 0
        self.train_epoch = 0
        self.test_step = 0
        self.test_epoch = 0

    def generate(self, which='train', augment=True, move_on=True):
        '''Output a single batch for training/testing.'''
        while True:
            if which=='train': scenes = self.train_scenes
            elif which=='test': scenes = self.test_scenes

            if which=='train':
                if self.train_step >= len(scenes):
                    self.train_step = 0
                    self.train_epoch += 1
                step = self.train_step
                epoch = self.train_epoch
            elif which=='test':
                if self.test_step >= len(scenes):
                    self.test_step = 0
                    self.test_epoch += 1
                step = self.test_step
                epoch = self.test_epoch

            n = min(self.batch_size, len(scenes)-step)
            X, Y, _, _ = self.get_batch(scenes[step:step+n])

            if augment: X, Y = self.augment(X, Y, epoch)
            if move_on:
                if which=='train':
                    self.train_step += n
                elif which=='test':
                    self.test_step += n

            yield X, Y

    def get_batch(self, scenes):
        '''Preprocess one batch of scenes and pass it as tensors.'''
        Xs = []
        Ys = []
        sms = []
        ids = []
        for scene in scenes:
            X = scene.get_data(upsample=self.upsample_input)
            Y = scene.hr
            Xs.append(X)
            Ys.append(Y)
            sms.append(scene.sm)
            ids.append(scene.id)
        
        Xs = self.images_to_tensor(np.array(Xs))
        Ys = self.images_to_tensor(np.array(Ys))
        sms = np.array(sms)
        ids = np.array(ids)
        return Xs, Ys, sms, ids

    def augment(self, X, Y, epoch):
        '''This augmentation is not random, rather we rotate 90 degrees CCW for each epoch passed.'''
        k = epoch % 4
        X = np.rot90(X, k, axes=(1,2))
        Y = np.rot90(Y, k, axes=(1,2))
        return X, Y

    def reset_for_training(self):
        '''This resets everything for training, in case you reuse one instance of this class 
        to train multiple models. This is not strictly necessary, but it's nice for deterministic 
        runs, as augmentation loops over on successive calls to generate().'''
        self.train_step = 0
        self.train_epoch = 0

    def reset_for_testing(self):
        '''This resets everything for testing.'''
        self.test_step = 0
        self.test_epoch = 0

    def load_scenes(self, path, red_range, nir_range, is_test=False):
        scenes = []
        if red_range:
            for i in range(*red_range):
                scene_path = path + 'RED/imgset{:04d}/'.format(i)
                scene = self.load_scene(scene_path, is_test=is_test)
                scenes.append(scene)
        if nir_range:
            for i in range(*nir_range):
                scene_path = path + 'NIR/imgset{:04d}/'.format(i)
                scene = self.load_scene(scene_path, is_test=is_test)
                scenes.append(scene)

        return scenes

    def load_scene(self, path, is_test=False):
        id = int(path[-5:-1])

        # Load low resolution images 
        lrs = []
        for i in range(19):
            filename = path + 'LR{:03d}.png'.format(i)
            if os.path.isfile(filename):
                img = self._load_image(filename)
                lrs.append(img)
            else:
                break

        # Load quality maps
        qms = []
        for i in range(19):
            filename = path + 'QM{:03d}.png'.format(i)
            if os.path.isfile(filename):
                img = self._load_image(filename, dtype=np.bool, convert_to_float=False)
                qms.append(img)
            else:
                break

        # Load status map
        sm = self._load_image(path + 'SM.png', dtype=np.bool, convert_to_float=False)

        # Load high res image if it's not test set
        hr = None
        if is_test==False:
            hr = self._load_image(path+'HR.png')
        return Scene(id, lrs, qms, sm, hr)

    def get_scene_path(self, scene_id):
        if scene_id in range(*self.train_red_range): path = self.train_path + '/RED/'
        elif scene_id in range(*self.train_nir_range): path = self.train_path + '/NIR/'
        elif scene_id in range(*self.test_red_range): path = self.test_path + '/RED/'
        elif scene_id in range(*self.test_nir_range): path = self.test_path + '/NIR/'
        else: raise ValueError('scene_id out of range:', scene_id)
        path += 'imgset{:04d}/'.format(scene_id)
        return path

    def score_image(self, sr, hr, sm, scene_id):
        path = self.get_scene_path(scene_id)
        return probav.embiggen.score_image(sr, path, (hr, sm))

    def score_images(self, sr_maps, highres_maps, status_maps, scene_ids):
        scores = []
        for i in range(sr_maps.shape[0]):
            sr = np.array(sr_maps[i])
            hr = np.array(highres_maps[i])
            sm = np.array(status_maps[i])
            score = self.score_image(sr, hr, sm, scene_ids[i])
            scores.append(score)
        scores = np.array([scores])
        return scores.mean()
        
    def _load_image(self, filename, dtype=np.uint16, convert_to_float=True):
        img = Image.open(filename)
        img = np.array(img, dtype=dtype)
        if convert_to_float: img = skimage.img_as_float(img)
        return img

    def images_to_tensor(self, images):
        if len(images.shape)<4:
            return images.reshape((*images.shape, 1))
        else:
            return images
    

class Scene():
    def __init__(self, id, lrs, qms, sm, hr=None):
        '''Scene class. Contains all data for a single scene: low res images, quality maps,
        status map, and high resolution image if available. Also has functions for preprocessing.
        '''
        self.id = id
        self.lrs = lrs
        self.qms = qms
        self.sm = sm
        self.hr = hr

    def get_data(self, upsample=False):
        '''The data that gets passed as input to the model.'''
        images = self.maximum_clearance()
        if upsample:
            images = self.baseline(images)
        return images

    def maximum_clearance(self, n=4):
        '''Stack n low resolution images that have maximum clearance.'''
        clearances = {}
        for i, qm in enumerate(self.qms):
            clearance = qm.sum()
            clearances[i] = clearance
        clearances = sorted(clearances.items(), key=lambda kv: kv[1], reverse=True)
        images = []
        for i in range(n):
            images.append(self.lrs[clearances[i][0]])
        image = np.stack(images, axis=2)
        return image

    def baseline(self, images, how='median'):
        '''Baseline implementation according to competition: Per pixel median followed by upsampling.'''
        images = self.central_tendency(images, how=how)
        images = skimage.transform.rescale(images, scale=3, order=5, mode='edge',
                                            anti_aliasing=False, multichannel=False)
        return images

    def central_tendency(self, images, how='median'):
        '''Simple averaging of pixel intensities.'''
        if how=='median':
            return np.median(images, axis=2)
        elif how=='mean':
            return np.mean(images, axis=2)

    def register(self, images, how='median', scale = 10.0):
        '''Perform image registration.'''
        # First image as reference
        ref_img = images[:,:,0]
        ref_img = skimage.transform.rescale(ref_img, scale=scale, order=5, mode='edge',
                                                anti_aliasing=False, multichannel=False)
        # Register subsequent images
        imgs = [ref_img]
        for i in range(1, images.shape[2]):
            img = images[:,:,i]
            shift, error, diffphase = register_translation(ref_img, img, 100)
            upsampled = skimage.transform.rescale(img, scale=scale, order=5, mode='edge',
                                                    anti_aliasing=False, multichannel=False)
            transform = skimage.transform.AffineTransform(translation=shift*scale)
            shifted = skimage.transform.warp(upsampled, transform, order=5, mode='edge', preserve_range=True)
            imgs.append(shifted)

        # Stack and upsample 3x
        imgs = np.stack(imgs, axis=2)
        img = self.central_tendency(images=imgs, how=how)
        img = skimage.transform.rescale(img, scale=3.0/scale, order=5, mode='edge',
                                        anti_aliasing=False, multichannel=False)
        return img

    def show(self, which='lr', **kwargs):
        if which=='hr':
            plt.imshow(self.hr, **kwargs)
            return
        elif which=='lr':
            imgs = self.lrs
        elif which=='qm':
            imgs = self.qms

        for i, img in enumerate(imgs):
            plt.subplot(5, 4, i+1)
            plt.imshow(img, **kwargs)
        plt.show()
            