#!/usr/bin/python
import os
import pathlib


from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import layers

datasets = ["medVar", "medVarObjBgColorBias", "medVarSpherBias", "medVarSpherObjColorBias",
            "medVarSphericalBgColorBias", "medVarBgBias", "medVarObjColorBias",
            "medVarTriple"]

results = pd.DataFrame(index=datasets, columns=datasets)


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

for model_name in datasets:
    checkpoint_filepath = os.path.join("models", "lenet_" + model_name)
    modernLenetModel.load_weights(checkpoint_filepath)

    for dataset_name in datasets:
        data_dir = os.path.join("datasets", dataset_name)
        data_dir = pathlib.Path(data_dir)

        test_dir = os.path.join(data_dir, "test")
        test_df = pd.read_json(os.path.join(test_dir, "parameters.jsonl"), lines=True)
        test_df["filename"] = test_df["id"] + ".png"

        datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = datagen.flow_from_dataframe(dataframe=test_df, directory=test_dir,
                                                     x_col="filename", y_col="obj_name",
                                                     batch_size=64)

        print("Evaljateing " + model_name + " on " + dataset_name)
        results.at[model_name, dataset_name] = modernLenetModel.evaluate(test_generator)[1]


results.to_csv("test_results.csv")
