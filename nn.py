import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.applications.xception import Xception
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train / 255., x_test / 255.
x_train = keras.preprocessing.image.smart_resize(x_train, (71, 71))
x_test = keras.preprocessing.image.smart_resize(x_test, (71, 71))
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

model_xception = Xception(weights='imagenet', include_top=False, input_shape=(71, 71, 3))
model_xception.trainable = False

model = Sequential()
model.add(model_xception)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

result = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.3)

