from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import  LeakyReLU
from keras.regularizers import l2
import numpy as np
from PIL import Image
import argparse
import math
import os
import time
from keras.utils import generic_utils
from scipy.misc import imsave
import matplotlib.pyplot as plt 
import cv2

img_w = 28
img_h = 28
img_c = 1
noise_dim = 100

epochs = 100
batch_size = 16
lr = 0.0002
weight_decay = 5e-4
# Read training data
#real_file = open("/home/zhaojian/Keras/MS_GAN/list/real.txt", "r")
#lines = real_file.readlines()
#real_file.close()
#N_real = len(lines)
#real_file_list = []
#for i in range(N_real):
#    real_file_list.append(lines[i].split()[0])
mnist = np.load('/home/zhaojian/Keras/MS_GAN/data/mnist.npz')
X_train = mnist['x_train']
X_train = (X_train - 127.5) / 127.5
X_train = X_train.reshape(60000, img_w, img_h, img_c)
N_real = 60000

def generator():
    
    mod_input = Input(shape=(noise_dim,))
    x = Dense(1024, activation = 'tanh')(mod_input)
    x = Dense(128*7*7, activation = 'tanh')(x)
    x = BatchNormalization()(x)
    x = Reshape((7, 7, 128))(x)    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (5, 5), activation = 'tanh', padding="same", kernel_regularizer=l2(weight_decay))(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(img_c, (5, 5), activation = 'tanh', padding="same", kernel_regularizer=l2(weight_decay))(x)
    
    return Model(mod_input, x)

def discriminator():
    
    mod_input = Input(shape = (img_w, img_h, img_c))
    x = Conv2D(64, (5, 5), activation = 'tanh', padding="same", kernel_regularizer=l2(weight_decay))(mod_input)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Conv2D(128, (5, 5), activation = 'tanh', padding="same", kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024, activation = 'tanh')(x) 
    x = Dense(1, activation = 'sigmoid')(x) 
    
    return Model(mod_input, x)

def gan(generator, discriminator):
    
    mod_input = generator.input
    x = generator(mod_input)
    x = discriminator(x)

    return Model(mod_input, x)

discriminator = discriminator()
generator = generator()
gan = gan(generator, discriminator)
#adam = Adam(lr=lr)
sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
generator.compile(loss='binary_crossentropy', optimizer=sgd)
gan.compile(loss='binary_crossentropy', optimizer=sgd)
discriminator.compile(loss='binary_crossentropy', optimizer=sgd)
noise = np.zeros((batch_size, noise_dim))
batches_per_epoch = int(N_real/batch_size)

# Training
for e in range(epochs):
    progbar = generic_utils.Progbar(batches_per_epoch*batch_size)
    start = time.time()
    for b in range(batches_per_epoch):        
        # Train discriminator
        discriminator.trainable = True
        noise = np.random.uniform(-1, 1, (batch_size, noise_dim))
       # real_img_file = real_file_list[b*batch_size:(b+1)*batch_size]
	#image_batch = np.zeros((batch_size, img_w, img_h, img_c), dtype=np.uint8)
       # for i in range(batch_size):
	#    real_img = Image.open(real_img_file[i])
         #   real_img = np.asarray(real_img, dtype="uint8" )
          #  real_img = (real_img - 127.5)/127.5
          #  real_img = real_img.reshape(1, img_w, img_h, img_c)
          #  image_batch[i, ...] = real_img
        image_batch = X_train[b*batch_size:(b+1)*batch_size, ...]
	generated_images = generator.predict(noise, verbose=0)
        X_d = np.concatenate([image_batch, generated_images], axis=0)
        Y_d = [1] * batch_size + [0] * batch_size
        d_loss = discriminator.train_on_batch(X_d, Y_d)
        # Train generator
        noise = np.random.uniform(-1, 1, (batch_size, noise_dim))
        discriminator.trainable = False
        X_g = noise
        Y_g = [1] * batch_size
        g_loss = gan.train_on_batch(X_g, Y_g)
        #Report results
        progbar.add(batch_size, values=[("Loss_D", d_loss),("Loss_G", g_loss)])
    print('\nEpoch {}/{}, Time: {}'.format(e + 1, epochs, time.time() - start))
    # Save generator and discriminator model every epoch
    generator.save_weights(('/home/zhaojian/Keras/MS_GAN/models/generator/'+'lr_'+str(lr)+'_epoch_'+str(e)+'_generator'), True)
    discriminator.save_weights(('/home/zhaojian/Keras/MS_GAN/models/discriminator/'+'lr_'+str(lr)+'_epoch_'+str(e)+'_discriminator'), True)
    # Save the internal result every epoch
    noise = np.random.uniform(-1, 1, (1, noise_dim))
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(img_w, img_h, img_c)
    generated_images = generated_images*127.5+127.5
    generated_images = (generated_images.squeeze()).astype(np.uint8)
    plt.imshow(generated_images)
    plt.savefig('/home/zhaojian/Keras/MS_GAN/internal_results/'+'lr_'+str(lr)+'_epoch_'+str(e)+'.jpg')
