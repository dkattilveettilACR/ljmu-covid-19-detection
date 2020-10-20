from __future__ import print_function, division
from tensorflow import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
import pandas as pd
import cv2

import matplotlib.pyplot as plt

import sys

import numpy as np
import os.path
from os import path
import scipy.misc
import argparse
import warnings
warnings.filterwarnings("ignore")

class GAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256

        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0001, 0.8)
        num_classes = 4

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator() #multi_gpu_model(self.build_discriminator())
        self.discriminator.compile(loss='binary_crossentropy',
                              optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator() #multi_gpu_model(self.build_generator())

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The GAN model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.gan = Model(z, valid)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(1024 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 1024)))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(512, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(256, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(128, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(64, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(32, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(1, kernel_size=3, strides=(1, 1), dilation_rate=2, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)


    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=(2, 2), input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=(2, 2), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=(2, 2), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=(2, 2), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=(2, 2), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(1024, kernel_size=3, strides=(2, 2), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)


    def load_xrays(self, epochs, batch_size=16, sample_interval=50):
        (img_x, img_y) = 256, 256
        metadata_csv = './gan_classifier/gan_data_tools/metadata.csv'
        dataTrain = pd.read_csv(metadata_csv)
        dataTrain = dataTrain[dataTrain['finding']==3]

        x_train = []
        # prepare label binarizer
        from sklearn import preprocessing

        count = 0
        for index, row in dataTrain.iterrows():
            img1 = row["filename"]
            if (path.exists(img1)):
                image1 = cv2.imread(img1)  # Image.open(img).convert('L')
                image1 = image1[:, :, 0]
                arr1 = cv2.resize(image1, (img_x, img_y))
                arr1 = arr1.astype('float32')
                arr1 /= 255.0
                arr1 = arr1 - np.mean(arr1)
                # DEBUG
                # print("shape of image: {}".format(arr1.shape))
                x_train.append(arr1)
                count += 1


            # DEBUG
        print("shape of x train: {}".format(len(x_train)))
        x_train = np.asarray(x_train)

        x_train = x_train.reshape(count, img_y, img_x, 1)

        valid1 = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid1)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.gan.train_on_batch(noise, valid1)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            #If at save interval => save generated image samples
            if (epoch+1) % sample_interval == 0:
                self.save_imgs(epoch+1)

    def save_imgs(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        #Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)   
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./data/generated/dcgan_covid/sample_%d.png" % epoch)
        plt.close()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'generate'], required=True, default = 'train')
    parser.add_argument("--checkpoint", type=str, required=False, default="./gan_classifier/model_weights/dcgan_covid/dcgen_covid.h5")
    parser.add_argument("--save", type=str, default = "./gan_classifier/model_weights/dcgan_covid/")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--image_count", type=int, default=100)
    parser.add_argument("--sample_interval", type=int, default=100)
    
    args = parser.parse_args()

    if args.mode == 'train':
        gan = GAN()
        gan.load_xrays(epochs=args.epochs, batch_size= args.bs, sample_interval = args.sample_interval)
        gan.generator.save(args.save + 'dcgen_covid.h5')
        gan.discriminator.save(args.save + 'dcdis_covid.h5')
    else :
        import math
        model = keras.models.load_model(args.checkpoint)
        # at the end, loop per class, per 1000 images
        cnt = args.image_count
        batch_count = int(math.ceil(cnt/10))
        for num in range(batch_count):
            noise1 = np.random.normal(0, 1, (10, 100))
            gen_imgs = model.predict(noise1)
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            if (cnt < 10):
                batch_images == cnt
            else 
                batch_images = 10
            cnt -= batch_images
            for i in range(batch_images):
                img = gen_imgs[i,:,:,0]
                img_index = i + num * 10
                scipy.misc.imsave("./data/generated/dcgan_covid/genxray_" + str(img_index)+".png", img)




