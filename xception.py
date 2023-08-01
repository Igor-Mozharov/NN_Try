import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.applications.convnext import ConvNeXtBase
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential

def preprocess_data(X, Y):
	X_p = keras.applications.xception.preprocess_input(X)
	Y_p = keras.utils.to_categorical(Y)
	return X_p, Y_p

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model_xception = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

inp = keras.Input(shape=(32, 32 ,3))
inp_resized = keras.layers.Lambda(lambda X: tf.image.resize(X, (299, 299)))(inp)

X = model_xception(inp_resized, training=False)
X = Flatten()(X)
X = Dense(500, activation='relu')(X)
X - Dropout(0.3)(X)
outputs = Dense(10, activation='softmax')(X)
model = keras.Model(inp, outputs)
model_xception.trainable = False
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# result = model.fit(x_train, y_train, batch_size=300, epochs=3, validation_data=(x_test, y_test))
print(model.summary())