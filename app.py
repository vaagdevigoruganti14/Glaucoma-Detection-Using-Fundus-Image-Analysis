!pip install tensorflow
import os
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# --------------------------------------------------
# Configuration
# --------------------------------------------------

base_dir = "data"

img_width = 224
img_height = 224

batch_size = 16
epochs = 20

model_filename = "glaucoma_vgg16_transfer_model.h5"

# --------------------------------------------------
# Dataset Paths
# --------------------------------------------------

train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# --------------------------------------------------
# Data Augmentation
# --------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# --------------------------------------------------
# Load Training Data
# --------------------------------------------------

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary"
)

# --------------------------------------------------
# Load Pretrained VGG16
# --------------------------------------------------

conv_base = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(img_width, img_height, 3)
)

# Freeze VGG16 layers
conv_base.trainable = False

# --------------------------------------------------
# Build Model
# --------------------------------------------------

model = Sequential()

model.add(conv_base)

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(
    256,
    activation="relu"
))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(
    1,
    activation="sigmoid"
))

# --------------------------------------------------
# Compile Model
# --------------------------------------------------

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(
        learning_rate=1e-4
    ),
    metrics=["accuracy"]
)

# --------------------------------------------------
# Model Summary
# --------------------------------------------------

model.summary()

# --------------------------------------------------
# Train Model
# --------------------------------------------------

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# --------------------------------------------------
# Save Model
# --------------------------------------------------

model.save(model_filename)

print(f"\nModel saved successfully as: {model_filename}")

# --------------------------------------------------
# Class Labels
# --------------------------------------------------

print("\nClass Indices:")
print(train_generator.class_indices)