#!/usr/bin/python
import os
import pathlib
import sys


from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

dataset_name = sys.argv[1]

data_dir = os.path.join("datasets", dataset_name)
data_dir = pathlib.Path(data_dir)

train_dir = os.path.join(data_dir, "train")
train_df = pd.read_json(os.path.join(train_dir, "parameters.jsonl"), lines=True)
train_df["filename"] = train_df["id"] + ".png"


valid_dir = os.path.join(data_dir, "validation")
valid_df = pd.read_json(os.path.join(valid_dir, "parameters.jsonl"), lines=True)
valid_df["filename"] = valid_df["id"] + ".png"


# Assuming that 1/255 is a good idea as it was with MNIST
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_dataframe(dataframe=train_df, directory=train_dir,
                                              x_col="filename", y_col="obj_name", batch_size=64)
valid_generator = datagen.flow_from_dataframe(dataframe=valid_df, directory=valid_dir,
                                              x_col="filename", y_col="obj_name", batch_size=64)
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

checkpoint_filepath = os.path.join("models", "lenet_" + str(dataset_name))
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False)

strategy = tf.distribute.OneDeviceStrategy("device:GPU:1")
with strategy.scope():
    modernLenetModel = keras.models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),
    ])
    #lenetModel.compile(optimizer="sgd", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
    modernLenetModel.compile(loss="categorical_crossentropy",
                             optimizer="adam", metrics=["accuracy"])


modernLenetModel.fit(train_generator,
                     steps_per_epoch=STEP_SIZE_TRAIN,
                     validation_data=valid_generator,
                     validation_steps=STEP_SIZE_VALID,
                     epochs=45,
                     callbacks = [model_checkpoint_callback]
                     )
