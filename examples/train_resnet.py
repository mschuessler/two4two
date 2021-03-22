import os
import pathlib


from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

data_dir = os.path.join("datasets", "high_variation")
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

checkpoint_filepath = os.path.join("models", "resnet50_highVariation")
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = keras.applications.resnet50.ResNet50(
        weights=None, pooling='avg', include_top=False, input_shape=(128, 128, 3))
    predictions = layers.Dense(2, activation='softmax')(base_model.output)
    resnet_model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    adam = keras.optimizers.Adam(lr=0.0003)
    resnet_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

resnet_model.fit(train_generator,
                 steps_per_epoch=STEP_SIZE_TRAIN,
                 validation_data=valid_generator,
                 validation_steps=STEP_SIZE_VALID,
                 epochs=30,
                 callbacks=[model_checkpoint_callback]
                 )
