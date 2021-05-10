#!/usr/bin/python
"""This script takes a name of dataset and trains a Modern LeNet model on that dataset.

The model is the evaluated against the other following datsets: spherical_color_bias, no_arms,
no_bias, spherical_bias and color_bias.
"""
import os
import pathlib
import sys


from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

dataset_name = sys.argv[1]

data_dir = os.path.join("two4two_datasets", dataset_name)
data_dir = pathlib.Path(data_dir)

train_dir = os.path.join(data_dir, "train")
train_df = pd.read_json(os.path.join(data_dir, "train", "parameters.jsonl"), lines=True)
train_df["filename"] = train_df["id"] + ".png"

valid_dir = os.path.join(data_dir, "validation")
valid_df = pd.read_json(os.path.join(valid_dir, "parameters.jsonl"), lines=True)
valid_df["filename"] = valid_df["id"] + ".png"

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_dataframe(dataframe=train_df, directory=train_dir,
                                              x_col="filename", y_col="label", batch_size=64)
valid_generator = datagen.flow_from_dataframe(dataframe=valid_df, directory=valid_dir,
                                              x_col="filename", y_col="label", batch_size=64)
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

model_filepath = "lenet_" + str(dataset_name)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=False)

strategy = tf.distribute.OneDeviceStrategy("device:GPU:0")

trained_model_exists = os.path.exists(model_filepath)
if trained_model_exists:
    modernLenetModel = keras.models.load_model(model_filepath)
else:
    with strategy.scope():
        modernLenetModel = keras.models.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
        ])
        modernLenetModel.compile(loss="categorical_crossentropy",
                                 optimizer="adam", metrics=["accuracy"])
    modernLenetModel.fit(train_generator,
                         steps_per_epoch=STEP_SIZE_TRAIN,
                         validation_data=valid_generator,
                         validation_steps=STEP_SIZE_VALID,
                         epochs=10,
                         callbacks=[model_checkpoint_callback]
                         )

datasets = ["spherical_color_bias", "no_arms", "no_bias", "spherical_bias", "color_bias"]
results = pd.DataFrame(index=datasets, columns=datasets)
for test_dataset_name in datasets:
    data_dir = os.path.join("two4two_datasets", test_dataset_name)
    data_dir = pathlib.Path(data_dir)

    test_dir = os.path.join(data_dir, "test")
    test_df = pd.read_json(os.path.join(test_dir, "parameters.jsonl"), lines=True)
    test_df["filename"] = test_df["id"] + ".png"

    datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen.flow_from_dataframe(dataframe=test_df, directory=test_dir,
                                                 x_col="filename", y_col="obj_name",
                                                 batch_size=64)

    print("Evaluating " + dataset_name + " on " + test_dataset_name)
    results.at[dataset_name, test_dataset_name] = modernLenetModel.evaluate(test_generator)[1]


results.to_csv("test_results.csv")
