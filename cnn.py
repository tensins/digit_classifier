import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from PIL import Image, ImageFilter
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import math
from keras.models import load_model
import copy as cp
from PIL import ImageEnhance as IE
from keras import regularizers

# using convolutional neural networks to classify digits 0-9 using keras
# and VGG11
# using: 3x3 kernels, dropout at each layer, 2x2 max pool, relu activation function

# we need to do the 28x28 to 32x32 padding because we're doing maxpool 5 times
# and at the final layer should get  1x1xd matrix and if we had 28x28, this would
# not be possible

n_batch_size = 128
n_epochs = 3

# callback to help us log the various losses and accuracy per batch
class LogCallback(Callback):
    
    def on_train_begin(self, logs={}):
        self.acc_log = []
        self.loss_log = []
        self.count = 0
        return

    def on_batch_end(self, batch, logs={}):
        self.acc_log.append(logs.get('acc'))
        self.loss_log.append(logs.get('loss'))
        self.count += 1
        return

# trains a model on the training data 
def vgg11(x_train, y_train, n_epochs, n_batch_size):
    # declare the model that will be our VGG11 CNN
    model = Sequential()

    # add a convolutional layer of 64 activation maps created by 3x3 kernels
    # have the padding so that the size of the image doesn't change
    # LAYER 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    # max pool with a 2x2 pool matrix
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))

    # another convlutional layer, this time of 128 depth
    # LAYER 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.10))

    # consecutive conv layers
    # LAYER 3
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.10))

    # LAYER 4
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.10))3

    # LAYER 5
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.10))
    
    # Fully connected layers
    # LAYER 6
    # flatten out the convolutions to get a layer of just neurons 
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.20))

    #LAYER 7
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.20))

    # LAYER 8
    model.add(Dense(1000, activation='relu'))
    #model.add(Dropout(0.20))

    model.add(Dense(10, activation='softmax'))
    #print(model.summary())

    # will be using categorical cross entropy as loss function
    # optimized via rmsprop 
    # for some reason, changing the learning rate to 0.0001 drastically improves performance
    rms = keras.optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    # metrics include accuracy and loss
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    
    train_log = LogCallback()
    # train the model with training data, it returns a history object from which we can analyse
    history = model.fit(x_train, y_train, batch_size=n_batch_size, epochs=n_epochs, callbacks=[train_log], verbose=1)
    
    
    return model, history, train_log

# vgg11 with l1 regularization
def vgg11_reg(x_train, y_train, n_epochs, n_batch_size):
    # declare the model that will be our VGG11 CNN
    model = Sequential()

    # add a convolutional layer of 64 activation maps created by 3x3 kernels
    # have the padding so that the size of the image doesn't change
    # LAYER 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    # max pool with a 2x2 pool matrix
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))

    # another convlutional layer, this time of 128 depth
    # LAYER 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.10))

    # consecutive conv layers
    # LAYER 3
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.10))

    # LAYER 4
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.10))3

    # LAYER 5
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.10))
    
    # Fully connected layers
    # LAYER 6
    # flatten out the convolutions to get a layer of just neurons 
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Dropout(0.20))

    #LAYER 7
    model.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    #model.add(Dropout(0.20))

    # LAYER 8
    model.add(Dense(1000, activation='relu'))
    #model.add(Dropout(0.20))

    model.add(Dense(10, activation='softmax'))
    #print(model.summary())

    # will be using categorical cross entropy as loss function
    # optimized via rmsprop 
    # for some reason, changing the learning rate to 0.0001 drastically improves performance
    rms = keras.optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
    # metrics include accuracy and loss
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    
    train_log = LogCallback()
    # train the model with training data, it returns a history object from which we can analyse
    history = model.fit(x_train, y_train, batch_size=n_batch_size, epochs=n_epochs, callbacks=[train_log], verbose=1)
    
    
    return model, history, train_log

# rotates the images in X by theta
# *** mutates the data
def img_rotate(X, theta):
    for i, x in enumerate(X):
        img = Image.fromarray(x).rotate(theta)
        # we shape because img.getdata() returns just 1D array of pixels
        X[i] = np.reshape(np.array(img.getdata()), (32, 32))


def img_blur(X, rad):
    for i, x in enumerate(X):
        img = Image.fromarray(x).filter(ImageFilter.GaussianBlur(rad))
        # we shape because img.getdata() returns just 1D array of pixels
        X[i] = np.reshape(np.array(img.getdata()), (32,32))

# augment an image :
# first perform gaussian blur with radius 1 to get rid of noise
# then perform round pixel intensities up to 255 if they're >= 128
# and down to 0 if they're < 128
def data_augment(X):
    for i, x in enumerate(X):
        img = Image.fromarray(x).filter(ImageFilter.GaussianBlur(1))
        img = img.convert('L')
        img = img.point(lambda p: 0 if p < 128 else 255, '1')
        # we shape because img.getdata() returns just 1D array of pixels
        X[i] = np.reshape(np.array(img.getdata()), (32,32))
        

# dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# padding training and test datapoints with 2 zeros on each
x_train = np.pad(x_train, ((0,0), (2,2), (2,2)), 'constant')
x_test = np.pad(x_test, ((0,0), (2,2), (2,2)), 'constant')

# currently, X values are uint8, want them to be floating point
# for accuracy
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# reshape to proper dimensions for convolution
x_train = np.reshape(x_train, (len(x_train), 32, 32, 1))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 1))


# reshape to proper dimensions for convolution
y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

# to avoid some categories being favored based on their classification number
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
'''
# Question 1.3a, image transformations
# rotating images:
# rotations to try: -45, -45, ... , 40, 45
rotations = np.arange(-45, 50, 5)
cnn_model = load_model('regularized_model.h5')
rotation_acc = []
for theta in rotations:
    x_test_copy = np.array(x_test, copy=True)
    x_test_copy = np.reshape(x_test_copy, (len(x_test_copy), 32, 32))
    x_test_copy = x_test_copy.astype('uint8')
    img_rotate(x_test_copy, theta)
    x_test_copy = np.reshape(x_test_copy, (len(x_test_copy), 32, 32, 1))
    x_test_copy = x_test_copy.astype('float32')
    score = cnn_model.evaluate(x_test_copy, y_test, batch_size=n_batch_size)
    rotation_acc.append(score[1])

plt.xscale('linear')
plt.plot(rotations, rotation_acc)
plt.title("Rotations vs Testing Accuracy")
plt.xlabel("Rotations")
plt.ylabel("Testing Accuracy")
plt.show()

# Question 1.3b
# blurring images:
# gaussian blur on image with radius 0, 1, 2, 3, 4, 5, 6

radii = np.arange(0, 7)
blur_acc = []
for rad in radii:
    x_test_copy = np.array(x_test, copy=True)
    x_test_copy = np.reshape(x_test_copy, (len(x_test_copy), 32, 32))
    # PIL images require uint8 pixels
    x_test_copy = x_test_copy.astype('uint8')
    # blur the image with radius rad
    img_blur(x_test_copy, rad)
    # change the type of each pixel intensity now to float32 for better precision
    x_test_copy = x_test_copy.astype('float32')
    x_test_copy = np.reshape(x_test_copy, (len(x_test_copy), 32, 32, 1))
    score = cnn_model.evaluate(x_test_copy, y_test, batch_size=n_batch_size)
    blur_acc.append(score[1])

plt.xscale('linear')
plt.plot(radii, blur_acc)
plt.title("Gaussian Blur Radius vs Testing Accuracy")
plt.xlabel("Gaussian Blur Radius")
plt.ylabel("Testing Accuracy")
plt.show()

'''
# should train a model with VGG11 configuration of layers
# augments the dataset
# have to change back to (32,32) image with 'uint8' so PIL cannot perform image transformation
#x_train = x_train.astype('uint8')
#x_train = np.reshape(x_train, (len(x_train), 32, 32))

#data_augment(x_train)

x_train = x_train.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 32, 32, 1))
cnn_model, history, train_logs = vgg11_reg(x_train[0:20000], y_train[0:20000], n_epochs, n_batch_size)
score = cnn_model.evaluate(x_test, y_test, batch_size=n_batch_size)
print(score)
cnn_model.save('regularized_model.h5')  # creates the augmented model
# run on test set, it returns a score list which contains the loss and accuracy
# unfortunately, evaluate doesn't take callbacks so must manually try each batch and get its accuracy/loss
#score = model.evaluate(x_test, y_test, batch_size=n_batch_size)
test_batches_x = np.array_split(x_test, math.ceil(len(x_test)*1.0/n_batch_size))
test_batches_y = np.array_split(y_test, math.ceil(len(y_test)*1.0/n_batch_size))
t_iterations = []
t_loss = []
t_acc = []
for i, x_batch in enumerate(test_batches_x):
    score = cnn_model.test_on_batch(x_batch, test_batches_y[i])
    t_iterations.append(i)
    t_loss.append(score[0])
    t_acc.append(score[1])



# plot training results
tr_iterations = np.arange(0, train_logs.count)

# plot loss
plt.plot(tr_iterations, train_logs.loss_log)
plt.title("Iterations vs Training Loss")
plt.xscale("log")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.show()

# plot accuracy
plt.plot(tr_iterations, train_logs.acc_log)
plt.title("Iterations vs Training Accuracy")
plt.xscale("log")
plt.xlabel("Iterations")
plt.ylabel("Training Accuracy")
plt.show()


# plot testing results

# plot loss
plt.plot(t_iterations, t_loss)
plt.title("Iterations vs Testing Loss")
plt.xscale("log")
plt.xlabel("Iterations")
plt.ylabel("Training Loss")
plt.show()

# plot accuracy
plt.plot(t_iterations, t_acc)
plt.title("Iterations vs Testing Accuracy")
plt.xscale("log")
plt.xlabel("Iterations")
plt.ylabel("Training Accuracy")
plt.show()
