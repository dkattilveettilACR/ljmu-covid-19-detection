from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model, model_from_json
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import pandas as pd
import cv2

import numpy as np
import os.path
from os import path

import numpy as np
import scipy.misc
import argparse
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, f1_score
import itertools
import matplotlib
import math
import seaborn as sn
from scipy import interp
from itertools import cycle
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
from tensorflow.python.client import device_lib

class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256

        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 4
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        optimizer_disc = Adam(0.00002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer_disc,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()
        n_nodes = 1024*8*8
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8, 8, 1024)))
        # upsample to 16x16
        model.add(Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 32x32
        model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 64x64
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 128x128
        model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 256x256
        model.add(Conv2DTranspose(32, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # output layer 256x256x1
        model.add(Conv2D(1, (5,5), activation='tanh', padding='same'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, (5,5), padding='same', input_shape=self.img_shape)) 
        model.add(LeakyReLU(alpha=0.2)) 
        # downsample to 128x128 
        model.add(Conv2D(64, (5,5), strides=(2,2), padding='same')) 
        model.add(LeakyReLU(alpha=0.2)) 
        # downsample to 64x64 
        model.add(Conv2D(128, (5,5), strides=(2,2), padding='same')) 
        model.add(LeakyReLU(alpha=0.2)) 
        # downsample to 32x32 
        model.add(Conv2D(256, (5,5), strides=(2,2), padding='same')) 
        model.add(LeakyReLU(alpha=0.2)) 
        # downsample to 16x16 
        model.add(Conv2D(512, (5,5), strides=(2,2), padding='same')) 
        model.add(LeakyReLU(alpha=0.2)) 
        # downsample to 8x8 
        model.add(Conv2D(1024, (5,5), strides=(2,2), padding='same')) 
        model.add(LeakyReLU(alpha=0.2)) 
        # classifier 
        model.add(Flatten()) 
        model.add(Dropout(0.4)) 

        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, sample_interval=1, equal_class=True):

        # Load the dataset
        (img_x, img_y) = 256, 256
        trainpath = './data/train.txt'
        dataTrain = pd.read_csv(trainpath, delimiter = ' ', names=['filename', 'finding'])
        
        if (equal_class == True):
            dataCovid = dataTrain[dataTrain['finding']==3]
            covidDataCount = len(dataCovid.index)
            dataNormal = dataTrain[dataTrain['finding']==0].sample(n = covidDataCount)
            dataBacterial = dataTrain[dataTrain['finding']==1].sample(n = covidDataCount)
            dataViral = dataTrain[dataTrain['finding']==2].sample(n = covidDataCount)

            dataTrain = pd.concat([dataNormal, dataBacterial, dataViral, dataCovid], axis = 0)

        print(dataTrain.info())

        x_train = []
        y_train = []
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
                label = row["finding"]
                y_train.append(label)
                count += 1

                # DEBUG
        print("shape of x train: {}".format(len(x_train)))
        x_train = np.asarray(x_train)
        x_train = x_train.reshape(count, img_y, img_x, 1)

        y_train = np.asarray(y_train)
        y_train = y_train.reshape(-1, 1)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        val_loss = 0
        old_val_loss = 100
        d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, 4, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-9 
            img_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_count*2, self.latent_dim))
            valid = np.ones((batch_count*2, 1))
            sampled_labels = np.random.randint(0, 4, (batch_count*2, 1))
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            # print average of real and fake [ training + validation loss, training accuracy, validation accuracy, generator loss
            metrics = self.discriminator.metrics_names
            print("Combined data performance: %d [%s: %f, %s: %.2f%%, %s: %.2f%%, , %s: %.2f%%, , %s: %.2f%%] [G loss: %f]"  
                   % (epoch, metrics[0], d_loss[0], metrics[1], d_loss[1], metrics[2], d_loss[2], 
                      metrics[3], 100*d_loss[3], metrics[4], 100*d_loss[4], g_loss[0]))
         
            #calculate validation loss and save best model
            val_loss = self.validate('./data/val.txt', batch_size)
            if (val_loss < old_val_loss):
                print("Old val loss: %.2f%%, new val loss: %.2f%%" % (old_val_loss, val_loss))
                self.save_model()
                old_val_loss = val_loss
            
            d1_hist.append(d_loss_real[0])
            d2_hist.append(d_loss_fake[0])
            g_hist.append(g_loss)
            a1_hist.append(d_loss_real[1])
            a2_hist.append(d_loss_fake[1])
                # If at save interval => save generated image samples
            if (epoch+1) % sample_interval == 0:
                self.sample_images(epoch+1)
        self.plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist)
      
    # create a line plot of loss for the gan and save to file
    def plot_history(self, d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	    # plot loss
        plt.subplot(2, 1, 1)
        plt.plot(d1_hist, label='d-real')
        plt.plot(d2_hist, label='d-fake')
        plt.plot(g_hist, label='gen')
        plt.legend()
	    # plot discriminator accuracy
        plt.subplot(2, 1, 2)
        plt.plot(a1_hist, label='acc-real')
        plt.plot(a2_hist, label='acc-fake')
        plt.legend()
	    # save plot to file
        plt.savefig('./gan_classifier/plots/plot_line_plot_loss.png')
        plt.close()

    def sample_images(self, epoch):
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./data/generated/dcgan_ac_covid/xrays_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "./gan_classifier/model_weights/dcgan_ac_covid/%s.json" % model_name
            weights_path = "./gan_classifier/model_weights/dcgan_ac_covid/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")

    def generate_arrays_from_dataframe(self, dataTest, batchsize):
        while True:
            # Load the dataset
            (img_x, img_y) = 256, 256

            x_test = []
            y_test = []
            # prepare label binarizer
            from sklearn import preprocessing
            batchcount = 0
            total_count = 0
            data_set_count = len(dataTest.index)
            for index, row in dataTest.iterrows():
                img1 = row[0]
                total_count += 1
                if (path.exists(img1)):
                    image1 = cv2.imread(img1)  # Image.open(img).convert('L')
                    image1 = image1[:, :, 0]
                    arr1 = cv2.resize(image1, (img_x, img_y))
                    arr1 = arr1.astype('float32')
                    arr1 /= 255.0
                    arr1 = arr1 - np.mean(arr1)
                    # DEBUG
                    # print("shape of image: {}".format(arr1.shape))
                    x_test.append(arr1)
                    label = row[1]
                    y_test.append(label)
                    batchcount += 1
                    if (batchcount == batchsize or total_count == data_set_count):
                        X_test = np.asarray(x_test)
                        X_test = X_test.reshape(batchcount, img_y, img_x, 1)
                        Y_test = np.asarray(y_test)
                        Y_test = Y_test.reshape(-1, 1)
                        valid = np.ones((batchcount, 1))
                        yield (X_test, [valid, Y_test])
                        x_test = []
                        y_test = []
                        batchcount = 0
                else:
                    print("file not found:", img1)

    def validate(self, path, batch_size=64):
        
        data_validation = pd.read_csv(path, delimiter = ' ', names=['filename', 'finding'])
        data_count = len(data_validation.index)
        steps = math.ceil(data_count/batch_size)
        generator = self.generate_arrays_from_dataframe(data_validation,batch_size)
        pred = self.discriminator.predict_generator(generator, steps =int(steps))
       
        gt = data_validation['finding'].to_numpy()
        val_loss = sparse_categorical_crossentropy(tf.convert_to_tensor(gt), tf.convert_to_tensor(pred[1]))
        val_loss_numpy = val_loss.eval()
        total_val_loss = np.sum(val_loss_numpy)/data_count
        return total_val_loss

        

    def evaluate(self, path, batch_size=64, cm_path='cm', roc_path='roc'):
        
        dataTest = pd.read_csv(path, delimiter = ' ', names=['filename', 'finding'])
        print(dataTest.info())
        datacount = len(dataTest.index)
        steps = math.ceil(datacount/batch_size)
        generator = self.generate_arrays_from_dataframe(dataTest,batch_size)
        pred = self.discriminator.predict_generator(generator, steps =int(steps))
        print('Confusion Matrix')
        labels = ["normal","bacterial","viral", "covid"]
        
        gt = []
        for index, row in dataTest.iterrows():
            gt_class = row[1]
            arr = np.zeros(4)
            arr[gt_class] =1
            gt.append(arr)

        gt = np.asarray(gt)
        self.compute_AUC_scores(gt, pred[1], labels)

        # Plot ROC scores
        self.plot_ROC_curve(gt, pred[1], labels, "./gan_classifier/plots/roc")

        # Treat the max. output as prediction. 
        # Plot Confusion Matrix
    
        pred = pred[1].argmax(axis=1)
        gt = gt.argmax(axis=1)
        self.plot_confusion_matrix(gt, pred, labels, "./gan_classifier/plots/cm")


    def plot_confusion_matrix(self, y_true, y_pred, labels, cm_path):
        norm_cm = confusion_matrix(y_true, y_pred, normalize='true')
        norm_df_cm = pd.DataFrame(norm_cm, index=labels, columns=labels)
        plt.figure(figsize = (10,10))
        sn.heatmap(norm_df_cm, annot=True, fmt='.2f', square=True, cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s_norm.png' % cm_path, pad_inches = 0, bbox_inches='tight')
        
        cm = confusion_matrix(y_true, y_pred)
        # Finding the annotations
        cm = cm.tolist()
        norm_cm = norm_cm.tolist()
        annot = [
            [("%d (%.2f)" % (c, nc)) for c, nc in zip(r, nr)]
            for r, nr in zip(cm, norm_cm)
        ]
        plt.figure(figsize = (10,7))
        sn.heatmap(norm_df_cm, annot=annot, fmt='', cbar=False, square=True, cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s.png' % cm_path, pad_inches = 0, bbox_inches='tight')
        print (cm)

        accuracy = np.sum(y_true == y_pred) / len(y_true)
        print ("Accuracy: %.5f" % accuracy)

    def compute_AUC_scores(self, y_true, y_pred, labels):
        """
        Computes the Area Under the Curve (AUC) from prediction scores

        y_true.shape  = [n_samples, n_classes]
        y_preds.shape = [n_samples, n_classes]
        labels.shape  = [n_classes]
        """
        AUROC_avg = roc_auc_score(y_true, y_pred)
        print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
        for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
            print('The AUROC of {0:} is {1:.4f}'.format(label, roc_auc_score(y, pred)))

    def plot_ROC_curve(self, y_true, y_pred, labels, roc_path): 
        """
        Plots the ROC curve from prediction scores

        y_true.shape  = [n_samples, n_classes]
        y_preds.shape = [n_samples, n_classes]
        labels.shape  = [n_classes]
        """
        n_classes = len(labels)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for y, pred, label in zip(y_true.transpose(), y_pred.transpose(), labels):
            fpr[label], tpr[label], _ = roc_curve(y, pred)
            roc_auc[label] = auc(fpr[label], tpr[label])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[label] for label in labels]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for label in labels:
            mean_tpr += interp(all_fpr, fpr[label], tpr[label])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.3f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.3f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=2)

        if len(labels) == 4:
            colors = ['green', 'cornflowerblue', 'darkorange', 'darkred']
        else:
            colors = ['green', 'cornflowerblue', 'darkred']
        for label, color in zip(labels, cycle(colors)):
            plt.plot(fpr[label], tpr[label], color=color, lw=lw,
                    label='ROC curve of {0} (area = {1:0.3f})'
                    ''.format(label, roc_auc[label]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        matplotlib.rcParams.update({'font.size': 14})
        plt.savefig('%s.png' % roc_path, pad_inches = 0, bbox_inches='tight')


if __name__ == '__main__':

    with tf.Session() as sess:
        print(device_lib.list_local_devices())
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=['train', 'evaluate', 'generate'], required=True, default = 'train')
        parser.add_argument("--checkpoint", type=str, required=False, default="./gan_classifier/model_weights/dcgan_ac_covid/discriminator_weights.hdf5")
        parser.add_argument("--save", type=str, default = "./gan_classifier/model_weights/")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--bs", type=int, default=10)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--image_count", type=int, default=100)
        parser.add_argument("--label", type=int, default=3)
        parser.add_argument("--sample_interval", type=int, default=50)
        parser.add_argument("--equal_class", type=bool, default=False)
    
        args = parser.parse_args()
        acgan = ACGAN()
        if args.mode == 'train':
            acgan.train(epochs=args.epochs, batch_size=args.bs, sample_interval = args.sample_interval, equal_class = args.equal_class)
        else:
            #json_file = open('./gan_classifier/model_weights/dcgan_ac_covid/discriminator.json', 'r')
            #loaded_model_json = json_file.read()
            #json_file.close()
            #loaded_model = model_from_json(loaded_model_json)
            ## load weights into new model
            #loaded_model.load_weights(args.checkpoint)
            acgan.discriminator.load_weights(args.checkpoint)
            if args.mode == 'evaluate':
               acgan.evaluate( path='./data/test.txt', batch_size = args.bs)
            else :
                # at the end, loop per class, per 1000 images
                cnt = args.image_count
                classes = {0:"normal", 1:"bacterial", 2:"viral", 3:"covid"}
                batch_count = int(cnt/10)
                for num in range(batch_count):
                    noise1 = np.random.normal(0, 1, (10, 100))
                    sampled_labels = np.array([args.label for _ in range(10)])
                    gen_imgs = acgan.discriminator.predict([noise, sampled_labels])
                    # Rescale images 0 - 1
                    gen_imgs = 0.5 * gen_imgs + 0.5
                    for i in range(10):
                        img = gen_imgs[i,:,:,0]
                        img_index = i + num * 10
                        scipy.misc.imsave("./data/generated/dcgan_ac_covid/xray_"+str(label)+"_" + str(img_index)+".png", img)

                #for label in range(0,4):
                #    r, c = 2, 2
                #    noise = np.random.normal(0, 1, (r * c, acgan.latent_dim))
                #    sampled_labels = np.array([label for _ in range(r) for num in range(c)])
                #    gen_imgs = loaded_model.predict([noise, sampled_labels])
                #    # Rescale images 0 - 1
                #    gen_imgs = 0.5 * gen_imgs + 0.5
                #    cnt = 0
                #    for i in range(r):
                #        for j in range(c):
                #            img = gen_imgs[cnt,:,:,0]
                #            scipy.misc.imsave("./data/generated/dcgan_ac_covid/class_" + classes[label] + str(cnt)+".png", img)
                #            cnt += 1
        