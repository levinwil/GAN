from keras import backend as K
K.set_image_dim_ordering('tf') # ensure our dimension notation matches
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import shutil
import glob
import os
from PIL import ImageOps

class GAN(object):
    '''
    GAN

    Parameters
    ____________
    img_width : Int
        the width of the images you will be classifying
    img_height : Int
        the height of the images you will be classifying
    GAN_name : String
        the name of the GAN
    Attributes
    ____________
    model : keras model
        The model after fitting
    img_width : Int
        the width of the images the model classifies
    img_height : Int
        hte height of the images the model classifies
    '''
    def __init__(self,
                 GAN_name,
                 img_width = 128,
                 img_height = 128):
        self.img_width = img_width
        self.img_height = img_height
        self.GAN_name = GAN_name

    '''
    generator_model

    Parameters
    ____________
    None

    Return
    ____________
    model : keras model
        The generator model
    '''
    def generator_model(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*8*8))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((8, 8, 128), input_shape=(128*8*8,)))
        model.add(UpSampling2D(size=(8, 8)))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        return model


    '''
    discriminator_model

    Parameters
    ____________
    None

    Return
    ____________
    model : keras model
        The discriminator model
    '''
    def discriminator_model(self):
        model = Sequential()
        model.add(
                Conv2D(64, (5, 5),
                padding='same',
                input_shape=(self.img_height, self.img_width, 1))
                )
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model

    '''
    generator_model
    combines the generator and discriminator models, freezing the discriminator
    model weights

    Parameters
    ____________
    g : keras model
        The generator model
    d :keras model
        The discriminator model

    Return
    ____________
    model : keras model
        The model of the generator followed by the discriminator
    '''
    def generator_containing_discriminator(self, g, d):
        model = Sequential()
        model.add(g)
        d.trainable = False
        model.add(d)
        return model

    '''
    load_data
    loads the data in the example_images directory, concatenating them into
    a numpy array

    Parameters
    ____________
    none

    Return
    ____________
    X_train : numpy array
        The concatenated training data
    '''
    def load_data(self):
        print("Loading data")
        X_train = []
        paths = glob.glob(os.path.normpath(os.getcwd() + '/example_images/*.jpg'))
        for path in paths:
            im = Image.open(path)
            im = ImageOps.fit(im, (self.img_height, self.img_width), Image.ANTIALIAS)
            im = ImageOps.grayscale(im)
            im = np.asarray(im)
            X_train.append(im)
        print("Finished loading data")
        return np.array(X_train)


    '''
    train
    trains a GAN model to the data in example_images, then saves the
    generator and discriminator models under the GAN_name in the weights
    directory

    Parameters
    ____________
    BATCH_SIZE : int
        the batch size you'd like to use
    NUM_EPOCHS : int
        the number of epochs you'd like to train for

    Return
    ____________
    void
    '''
    def train(self, BATCH_SIZE, NUM_EPOCHS = 1000):
        X_train = self.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = X_train[:, :, :, None]
        d = self.discriminator_model()
        g = self.generator_model()
        d_on_g = self.generator_containing_discriminator(g, d)
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g.compile(loss='binary_crossentropy', optimizer="SGD")
        d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
        d.trainable = True
        d.compile(loss='binary_crossentropy', optimizer=d_optim)
        shutil.rmtree('generated_images')
        os.makedirs('generated_images')
        for epoch in range(NUM_EPOCHS):
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
            for index in range(int(X_train.shape[0]/BATCH_SIZE)):
                noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
                image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                generated_images = g.predict(noise, verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                d_loss = d.train_on_batch(X, y)
                print("batch %d d_loss : %f" % (index, d_loss))
                noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
                d.trainable = False
                g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
                d.trainable = True
                print("batch %d g_loss : %f" % (index, g_loss))
                if index % 10 == 1:
                    g.save_weights('weights/' + str(self.GAN_name) + '_generator', True)
                    d.save_weights('weights/' + str(self.GAN_name) + '_discriminator', True)

    '''
    generate
    generates NUM_PICTURES_TO_GENERATE images using the generator model and
    saves those images to the generated_images directory

    Parameters
    ____________
    NUM_PICTURES_TO_GENERATE : int
        the number of pictures you'd like to generate
    nice : int
        if you'd like to do some post-processing on the image to clean it up

    Return
    ____________
    void
    '''
    def generate(self, NUM_PICTURES_TO_GENERATE = 1, nice = False):
        g = self.generator_model()
        g.compile(loss='binary_crossentropy', optimizer="SGD")
        g.load_weights('weights/' + str(self.GAN_name) + '_generator')
        if nice:
            d = self.discriminator_model()
            d.compile(loss='binary_crossentropy', optimizer="SGD")
            d.load_weights('weights/' + str(self.GAN_name) + '_discriminator')
            noise = np.random.uniform(-1, 1, (NUM_PICTURES_TO_GENERATE*20, 100))
            generated_images = g.predict(noise, verbose=1)
            d_pret = d.predict(generated_images, verbose=1)
            index = np.arange(0, NUM_PICTURES_TO_GENERATE*20)
            index.resize((NUM_PICTURES_TO_GENERATE*20, 1))
            pre_with_index = list(np.append(d_pret, index, axis=1))
            pre_with_index.sort(key=lambda x: x[0], reverse=True)
            nice_images = np.zeros((NUM_PICTURES_TO_GENERATE,) \
            + generated_images.shape[1:3], dtype=np.float32)
            nice_images = nice_images[:, :, :, None]
            for i in range(NUM_PICTURES_TO_GENERATE):
                idx = int(pre_with_index[i][1])
                nice_image = generated_images[idx, :, :, 0]*127.5 + 127.5
                Image.fromarray(nice_image.astype(np.uint8)).save(
                    "generated_images/" + str(self.GAN_name) + "_generated_image_" + str(i) + ".png")
        else:
            noise = np.random.uniform(-1, 1, (NUM_PICTURES_TO_GENERATE, 100))
            generated_images = g.predict(noise, verbose=1)
            for i in range(NUM_PICTURES_TO_GENERATE):
                generated_image = generated_images[i, :, :, 0]*127.5 + 127.5
                Image.fromarray(generated_image.astype(np.uint8)).save(
                    "generated_images/" + str(self.GAN_name) + "_generated_image_" + str(i) + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A general image generator.')
    parser.add_argument("GAN_name", help="The name of the GAN",
                    type=str)
    parser.add_argument("--train", help="Fit a model to the data in \
    example_images, and save the generator and discriminator. Type true after \
    if you'd like to train.")
    parser.add_argument("--generate", help="Generate data using the generator \
    under the GAN name. Saves the data to generated_images. Type true after if \
    you'd like to generate.")
    args = parser.parse_args()

    if args.GAN_name == "" or args.GAN_name == None :
        raise ValueError("You must specify a GAN name")

    gan = GAN(GAN_name = args.GAN_name, img_width = 128, img_height = 128)
    if args.train:
        gan.train(BATCH_SIZE = 10, NUM_EPOCHS = 1000)
    if args.generate:
        gan.generate(NUM_PICTURES_TO_GENERATE = 1, nice = True)
