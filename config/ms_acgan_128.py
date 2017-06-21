#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import cv2
from keras.layers import Deconvolution2D

np.random.seed(1337)

K.set_image_dim_ordering('th')

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 128, 128)
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
    cnn.add(Dense(12 * 8 * 8, activation='relu'))
    cnn.add(Reshape((12, 8, 8)))

    # upsample to (..., 16, 16)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(384, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    #cnn.add(BatchNormalization())

    # upsample to (..., 32, 32)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(256, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    #cnn.add(BatchNormalization())

    # upsample to (..., 64, 64)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(192, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_normal'))
    #cnn.add(BatchNormalization())

    # upsample to (..., 128,128)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Conv2D(3, (5, 5), padding='same', activation='relu', kernel_initializer='glorot_normal'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, latent_size,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)

def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()

    cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), border_mode='same',
                          input_shape=(3, 128, 128)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), border_mode='same'))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    #cnn.add(BatchNormalization())

    cnn.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), border_mode='same'))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    #cnn.add(BatchNormalization())

    cnn.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), border_mode='same'))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    #cnn.add(BatchNormalization())

    cnn.add(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), border_mode='same'))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    #cnn.add(BatchNormalization())

    cnn.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), border_mode='same'))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))
    #cnn.add(BatchNormalization())

    cnn.add(Flatten())

    image = Input(shape=(3, 128, 128))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])

if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 800
    batch_size = 2
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.00002
    adam_beta_1 = 0.5
    #adam_beta_2 = 0.999

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, image_class], output=[fake, aux])

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )

    # get our mnist data, and force it to be of shape (..., 1, 28, 28) with
    # range [-1, 1]
    train_file = open("/home/zhaojian/Keras/MS_GAN/list/train_10.txt", "r")
    train_lines = train_file.readlines()
    train_file.close()
    N_train = len(train_lines)
    train_file_list = []
    y_train = np.zeros([N_train], dtype=np.uint8)
    for i in range(N_train):
        train_file_list.append(train_lines[i].split()[0])
        y_train[i] = int(train_lines[i].split()[1])
    X_train = np.zeros((N_train, 128, 128, 3), dtype=np.uint8)
    for k in range(N_train):
        Real_img = cv2.resize(cv2.imread(train_file_list[k]), (128, 128)).astype(np.uint8)
        X_train[k, ...] = Real_img
    X_train = X_train.squeeze().transpose(0, 3, 1, 2)

    test_file = open("/home/zhaojian/Keras/MS_GAN/list/test_10.txt", "r")
    test_lines = test_file.readlines()
    train_file.close()
    N_test = len(test_lines)
    test_file_list = []
    y_test = np.zeros([N_test], dtype=np.uint8)
    for i in range(N_test):
        test_file_list.append(test_lines[i].split()[0])
        y_test[i] = int(test_lines[i].split()[1])
    X_test = np.zeros((N_test, 128, 128, 3), dtype=np.uint8)
    for k in range(N_test):
        Real_img = cv2.resize(cv2.imread(test_file_list[k]), (128, 128)).astype(np.uint8)
        X_test[k, ...] = Real_img
    X_test = X_test.squeeze().transpose(0, 3, 1, 2)

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X_train.shape[0] / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            progress_bar.update(index)
            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, 10, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * batch_size + [0] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, 10, 2 * batch_size)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, 10, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, 10, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            '/home/zhaojian/Keras/MS_GAN/models/generator/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            '/home/zhaojian/Keras/MS_GAN/models/discriminator/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, latent_size))

        sampled_labels = np.array([
            [i] * 10 for i in range(10)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)
        generated_images.transpose(0, 2, 3, 1)
        generated_images_r = generated_images.transpose(0, 2, 3, 1)[:,:,:,0]
        generated_images_g = generated_images.transpose(0, 2, 3, 1)[:,:,:,1]
        generated_images_b = generated_images.transpose(0, 2, 3, 1)[:,:,:,2]
        # arrange them into a grid
        img_r = (np.concatenate([r.reshape(-1, 128)
                               for r in np.split(generated_images_r, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
        img_g = (np.concatenate([r.reshape(-1, 128)
                               for r in np.split(generated_images_g, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
        img_b = (np.concatenate([r.reshape(-1, 128)
                               for r in np.split(generated_images_b, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
        img = np.zeros((1280, 1280, 3), dtype=np.uint8)
        img[:,:,0] = img_r
        img[:,:,1] = img_g
        img[:,:,2] = img_b

        Image.fromarray(img[:,:,np.array([2,1,0])]).save(
            '/home/zhaojian/Keras/MS_GAN/internal_results/plot_epoch_{0:03d}_generated.png'.format(epoch))

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))
