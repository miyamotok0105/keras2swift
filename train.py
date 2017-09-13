# coding: utf-8
import os
import glob
import numpy as np
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

path = "./font"
files = os.listdir(path)
files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]

x_train = []
y_train = []
y_list = []

for d in files_dir:
  y_list.append(len(y_list)-1)
  filelist = glob.glob('font/%s/*.png'%(d))
  for f in filelist:
    x_train.append(np.array(Image.open(f).resize((100, 100))))
    y_train.append(len(y_list)-1)

y_train = np.array(y_train)
x_train = np.array(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes=9)

print(x_train.shape)
print(y_train.shape)


# x = np.array([np.array(Image.open(fname).resize((256, 256))) for fname in filelist])

# # 疑似データ生成
# x_train = np.random.random((100, 100, 100, 3))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# x_test = np.random.random((20, 100, 100, 3))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# print(x_train.shape)
# print(y_train.shape)

model = Sequential()
# 入力: サイズが100x100で3チャンネルをもつ画像 -> (100, 100, 3) のテンソル
# それぞれのlayerで3x3の畳み込み処理を適用している
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=3)
model.save('model.h5')
# score = model.evaluate(x_test, y_test, batch_size=32)



import coremltools
coreml_model = coremltools.converters.keras.convert(model)
coreml_model.save("keras_model.mlmodel")








# from __future__ import print_function
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten, Input
# from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import np_utils
# from keras.layers.normalization import BatchNormalization
# import sys

# import numpy as np
# from keras.datasets import mnist

# batch_size = 32
# nb_classes = 5
# nb_epoch = 200
# data_augmentation = True

# img_rows, img_cols = 112, 112
# img_channels = 3

# # X_train = [[1,1,0], [1,1,0], [1,1,0]]
# # y_train = [1, 1, 1]
# # X_test = [[1,1,0], [1,1,0], [1,1,0]]
# # y_test = [1, 1, 1]

# # X_train = np.array([[1,1,0], [1,1,0], [1,1,0]])
# # y_train = np.array([1, 1, 1])
# # X_test = np.array([[1,1,0], [1,1,0], [1,1,0]])
# # y_test = np.array([1, 1, 1])

# X_train = np.arange(983040).reshape((5,256,256,3))
# y_train = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
# X_test = np.arange(983040).reshape((5,256,256,3))
# y_test = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])


# print('X_train shape:', X_train.shape)
# print('y_train shape:', y_train.shape)
# # print(X_train.shape[0], 'train samples')
# # print(X_test.shape[0], 'test samples')

# # Y_train = np_utils.to_categorical(y_train, nb_classes)
# # Y_test = np_utils.to_categorical(y_test, nb_classes)

# model = Sequential()

# model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
# csv_logger = CSVLogger('log.csv', append=True, separator=';')

# fpath = 'weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.h5'
# cp_cb = ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# if not data_augmentation:
#     print('Not using data augmentation.')
#     model.fit(X_train, Y_train,
#               batch_size=batch_size,
#               nb_epoch=nb_epoch,
#               validation_data=(X_test, Y_test),
#               shuffle=True,
#               callbacks=[csv_logger, cp_cb, stopping])
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images

#     # Compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(X_train)

#     # Fit the model on the batches generated by datagen.flow().
#     model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
#                         steps_per_epoch=len(X_train),
#                         epochs=nb_epoch,
#                         validation_data=(X_test, Y_test),
#                         callbacks=[csv_logger])

# model.save('model.h5')
