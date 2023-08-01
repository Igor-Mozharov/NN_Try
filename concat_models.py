import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import cifar10
    from keras.applications.densenet import DenseNet169
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Activation, Conv2D, Reshape, Input, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adamax
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers.experimental.preprocessing import RandomRotation, RandomTranslation, RandomFlip, RandomContrast, RandomHeight, RandomWidth
from keras.

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train / 255., x_test / 255.
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# model_den = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# model_den.trainable = False
# print(model_den.summary())
#
# model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# model_vgg.trainable = False

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), kernel_regularizer='l2'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer='l2'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), kernel_regularizer='l2'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer='l2'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3), kernel_regularizer='l2'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer='l2'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])