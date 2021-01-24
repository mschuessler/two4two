%matplotlib inline
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
data_dir = pathlib.Path(data_dir)

train_df = pd.read_json(os.path.join(data_dir,"labels_train.json"), lines=True)
train_df['obj_type'] = train_df['obj_type'].astype(str)

test_df = pd.read_json(os.path.join(data_dir,"labels_test.json"), lines=True)
test_df['obj_type'] = test_df['obj_type'].astype(str)
# Assuming that 1/255 is a good idea as it was with MNIST
datagen=ImageDataGenerator(rescale=1./255)
train_generator=datagen.flow_from_dataframe(dataframe=train_df, directory=data_dir, x_col="filename", y_col="obj_type", batch_size=128, color_mode="grayscale")
valid_generator=datagen.flow_from_dataframe(dataframe=test_df, directory=data_dir, x_col="filename", y_col="obj_type", batch_size=128, color_mode="grayscale")
modernLenetModel = keras.models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),
        ])
#lenetModel.compile(optimizer="sgd", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
modernLenetModel.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
modernLenetModel.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25,
                    callbacks = [PlotLossesKerasTF()])
