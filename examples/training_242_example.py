import tensorflow as tf
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from livelossplot import PlotLossesKerasTF
import numpy as np
import pandas as pd
import json
import pathlib
import PIL
import PIL.Image
import os




dataset_url = "https://f001.backblazeb2.com/file/mschuessler-share/two4two_leon_dataset/obj_color_bias.tar.gz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, fname='obj_color_bias', untar=True)
data_dir = os.path.join("data","color_bias")

data_dir = pathlib.Path(data_dir)

train_df = pd.read_json(os.path.join(data_dir,"labels_train.json"), lines=True)
train_df['obj_type'] = train_df['obj_type'].astype(str)

# new way of loading
test_dir = os.path.join(data_dir,"test")
test_df = pd.read_json(os.path.join(test_dir,"parameters.jsonl"), lines=True)
test_df["filename"] = test_df["id"] + ".png"

#test_df['obj_type'] = test_df['obj_type'].astype(str)

test_df
# Assuming that 1/255 is a good idea as it was with MNIST
datagen=ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_dataframe(dataframe=test_df, directory=test_dir, x_col="filename", y_col="obj_name", batch_size=64)
valid_generator=datagen.flow_from_dataframe(dataframe=test_df, directory=data_dir, x_col="filename", y_col="obj_type", batch_size=64)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    modernLenetModel = keras.models.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            #layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            #layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            #layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            #layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
            ])
    #lenetModel.compile(optimizer="sgd", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
    modernLenetModel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



modernLenetModel.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    #validation_data=valid_generator,
                    #validation_steps=STEP_SIZE_VALID,
                    epochs=30,
                    #callbacks = [PlotLossesKerasTF()]
                    )
# grayscale + flat + 5 epochs = .83
# color + flat + 4 epochs = .86
# + BatchNormalization .83
# depper + BatchNormalization .87
# just deeper: .87

modernLenetModel.summary()


class Residual(keras.Model):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = layers.Conv2D(num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels, kernel_size=1, strides=strides)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, X):
        Y = keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return keras.activations.relu(Y)

class ResnetBlock(layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,**kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X

with strategy.scope():
    resnetModel = tf.keras.Sequential([
            # The following layers are the same as b1 that we created earlier
            layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            # The following layers are the same as b2, b3, b4, and b5 that we
            # created earlier
            ResnetBlock(64, 2, first_block=True),
            ResnetBlock(128, 2),
            ResnetBlock(256, 2),
            ResnetBlock(512, 2),
            layers.GlobalAvgPool2D(),
            layers.Dense(units=2)])
    resnetModel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

resnetModel.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=30,
                    #callbacks = [PlotLossesKerasTF()]
                    )

def vgg_block(num_convs, num_channels):
    blk = keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(layers.Conv2D(num_channels,kernel_size=3,padding='same',activation='relu'))

    blk.add(layers.MaxPool2D(pool_size=2, strides=2))
    return blk

# This is the original VGG Size
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
ratio = 2

#This is the modified sizie
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]

with strategy.scope():
    vgg11Model = keras.models.Sequential()
    # The convulational part
    for (num_convs, num_channels) in small_conv_arch:
        vgg11Model.add(vgg_block(num_convs, num_channels))
    # The fully-connected part
    vgg11Model.add(keras.models.Sequential([
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2)]))

    vgg11Model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

vgg11Model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=30,
                    callbacks = [PlotLossesKerasTF()]
                    )
