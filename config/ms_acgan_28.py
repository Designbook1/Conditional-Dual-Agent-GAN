#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function

from collections import defaultdict
import cPickle as pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
import cv2
import argparse,os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

experiment_name = "06192017_each_100/"


np.random.seed(1337)

K.set_image_dim_ordering('th')



def build_generator(latent_size):
	# we will map a pair of (z, L), where z is a latent vector and L is a
	# label drawn from P_c, to image space (..., 1, 28, 28)
	cnn = Sequential()

	cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
	cnn.add(Dense(128 * 7 * 7, activation='relu'))
	cnn.add(Reshape((128, 7, 7)))

	# upsample to (..., 14, 14)
	cnn.add(UpSampling2D(size=(2, 2)))
	cnn.add(Convolution2D(256, 5, 5, border_mode='same',
						  activation='relu', init='glorot_normal'))

	# upsample to (..., 28, 28)
	cnn.add(UpSampling2D(size=(2, 2)))
	cnn.add(Convolution2D(128, 5, 5, border_mode='same',
						  activation='relu', init='glorot_normal'))

	# take a channel axis reduction
	cnn.add(Convolution2D(3, 2, 2, border_mode='same',
						  activation='tanh', init='glorot_normal'))

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

	cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
						  input_shape=(3, 28, 28)))
	cnn.add(LeakyReLU())
	cnn.add(Dropout(0.3))

	cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
	cnn.add(LeakyReLU())
	cnn.add(Dropout(0.3))

	cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
	cnn.add(LeakyReLU())
	cnn.add(Dropout(0.3))

	cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
	cnn.add(LeakyReLU())
	cnn.add(Dropout(0.3))

	cnn.add(Flatten())

	image = Input(shape=(3, 28, 28))

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
	nb_epochs = 10000
	batch_size = 2
	latent_size = 100

	# Adam parameters suggested in https://arxiv.org/abs/1511.06434
	adam_lr = 0.00002
	adam_beta_1 = 0.5

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
# -------------------------------------------------------------------------
	train_file = open("../list/10train.txt", "r")
	# train_file = open("../list/10train_Ext.txt", "r")
	# train_file = open("../list/10_train_celeb_cropped.txt", "r")

	train_lines = train_file.readlines()
	# train_lines = train_file.readlines()[:100]

	train_file.close()
	N_train = len(train_lines)
	train_file_list = []
	y_train = np.zeros([N_train], dtype=np.uint8)
	for i in range(N_train):
		train_file_list.append(train_lines[i].split("\t")[0])
		y_train[i] = int(train_lines[i].split("\t")[1])
		print "path ", train_lines[i].split("\t")[0]
		print "label: ", int(train_lines[i].split("\t")[1])

	X_train = np.zeros((N_train, 28, 28, 3), dtype=np.uint8)
	for k in range(N_train):
		print "train_file_list[k]: ", train_file_list[k]
		Real_img = cv2.resize(cv2.imread(train_file_list[k]), (28, 28)).astype(np.uint8)
		print "Real_img.shape: ", Real_img.shape
		# Real_img = Real_img.copy()
		# Real_img[:, :, 0] = (Real_img[:, :, 0] - 103.94)/103.94
		# Real_img[:, :, 1] = (Real_img[:, :, 1] - 116.78)/116.78
		# Real_img[:, :, 2] = (Real_img[:, :, 2] - 123.68)/123.68
		X_train[k, ...] = Real_img

	# b_mean = float(np.mean(X_train[:, :, :, 0]))
	# g_mean = float(np.mean(X_train[:, :, :, 1]))
	# r_mean = float(np.mean(X_train[:, :, :, 2]))

	# (X_train[:, :, :, 0] - b_mean)/b_mean
	# (X_train[:, :, :, 1] - g_mean)/g_mean
	# (X_train[:, :, :, 2] - r_mean)/r_mean

	print "X_train.shape: ", X_train.shape

	X_train = X_train.squeeze().transpose(0, 3, 1, 2)

	test_file = open("../list/10test.txt", "r")
	# test_file = open("../list/10test_Ext.txt", "r")
	# test_file = open("../list/10_test_celeb_cropped.txt", "r")


	test_lines = test_file.readlines()
	train_file.close()
	N_test = len(test_lines)
	test_file_list = []
	y_test = np.zeros([N_test], dtype=np.uint8)
	for i in range(N_test):
		test_file_list.append(test_lines[i].split("\t")[0])
		y_test[i] = int(test_lines[i].split("\t")[1])
	X_test = np.zeros((N_test, 28, 28, 3), dtype=np.uint8)
	for k in range(N_test):
		Real_img = cv2.resize(cv2.imread(test_file_list[k]), (28, 28)).astype(np.uint8)
		print "Real_img.shape: ", Real_img.shape
		# Real_img = Real_img.copy()
		# Real_img[:, :, 0] = (Real_img[:, :, 0] - 103.94)/103.94
		# Real_img[:, :, 1] = (Real_img[:, :, 1] - 116.78)/116.78
		# Real_img[:, :, 2] = (Real_img[:, :, 2] - 123.68)/123.68
		X_test[k, ...] = Real_img

	print "X_test.shape: ", X_test.shape
	print "X_test[:, :, :, 0].shape: ", X_test[:, :, :, 0].shape

	# b_mean = float(mean(X_test[:, :, :, 0]))
	# g_mean = float(mean(X_test[:, :, :, 1]))
	# r_mean = float(mean(X_test[:, :, :, 2]))

	# (X_test[:, :, :, 0] - b_mean)/b_mean
	# (X_test[:, :, :, 1] - g_mean)/g_mean
	# (X_test[:, :, :, 2] - r_mean)/r_mean

	X_test = X_test.squeeze().transpose(0, 3, 1, 2)
# -------------------------------------------------------------------------
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
		path = '../models/generator/' + experiment_name
		if os.path.exists(path) == False:
				os.mkdir(path)
		generator.save_weights(
			path + '/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)

		path = '../models/discriminator/' + experiment_name
		if os.path.exists(path) == False:
				os.mkdir(path)
		discriminator.save_weights(
			path + '/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

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

		img_r = (np.concatenate([r.reshape(-1, 28)
							   for r in np.split(generated_images_r, 10)
							   ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
		img_g = (np.concatenate([r.reshape(-1, 28)
							   for r in np.split(generated_images_g, 10)
							   ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
		img_b = (np.concatenate([r.reshape(-1, 28)
							   for r in np.split(generated_images_b, 10)
							   ], axis=-1) * 127.5 + 127.5).astype(np.uint8)


		# img_r = (np.concatenate([r.reshape(-1, 28)
		#                        for r in np.split(generated_images_r, 10)
		#                        ], axis=-1) * r_mean + r_mean).astype(np.uint8)
		# img_g = (np.concatenate([r.reshape(-1, 28)
		#                        for r in np.split(generated_images_g, 10)
		#                        ], axis=-1) * g_mean + g_mean).astype(np.uint8)
		# img_b = (np.concatenate([r.reshape(-1, 28)
		#                        for r in np.split(generated_images_b, 10)
		#                        ], axis=-1) * b_mean + b_mean).astype(np.uint8)



		img = np.zeros((280, 280, 3), dtype=np.uint8)
		img[:,:,0] = img_r
		img[:,:,1] = img_g
		img[:,:,2] = img_b

		path = '../internal_results/' + experiment_name
		if os.path.exists(path) == False:
				os.mkdir(path)

		print "Image.fromarray(img[:,:,np.array([2,1,0])]).size: "
		print Image.fromarray(img[:,:,np.array([2,1,0])]).size

		Image.fromarray(img[:,:,np.array([2,1,0])]).save(
			path + '/plot_epoch_{0:03d}_generated.png'.format(epoch))

		print "np.asarray(img[:,:,np.array([2,1,0])])[:10]"
		print np.asarray(img[:,:,np.array([2,1,0])])[:10]

	# path = '../log/' + experiment_name
	# if os.path.exists(path) == False:
	# 		os.mkdir(path)

	pickle.dump({'train': train_history, 'test': test_history},
				open(path + '/acgan-history.pkl', 'wb'))
