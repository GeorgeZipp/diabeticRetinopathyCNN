import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 26
img_width = 26
batch_size = 4

activation = 'sigmoid'
model = keras.Sequential([
    layers.Input((26,26,1)),
    #layers.Conv2D(16,3, activation=activation,padding='same'),
    #layers.BatchNormalization(),
    layers.Conv2D(16,3, activation=activation,padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    #layers.Conv2D(32, 3, activation=activation,padding ='same'),
    #layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation=activation,padding ='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation=activation, kernel_initializer='he_uniform'),
    layers.Dense(3, activation='softmax')
])
model.summary()
#load data using dataset_from_directory
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'patches/subpatches/',
    labels='inferred',
    label_mode = "categorical",
    color_mode='grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle=True,
    seed=456,
    validation_split=0.1,
    subset="training"
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'patches/subpatches/',
    labels='inferred',
    label_mode = "categorical",
    color_mode='grayscale',
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle=True,
    seed=456,
    validation_split=0.1,
    subset="validation"
)


model.compile(
    optimizer='rmsprop',
    loss = 'categorical_crossentropy',
    metrics=["accuracy"],
)

#train_datagen = ImageDataGenerator(rotation_range=45,
#                                   width_shift_range=0.2,
#                                   zoom_range=0.2,
#                                   horizontal_flip=True)
#train_datagen.fit(ds_train)
#train_generator = train_datagen.flow(
#    ds_train,
#    batch_size = 32
#)
#modelplot = model.fit_generator(
#    train_generator,
#    steps_per_epoch =1000,
#    epochs = 20,
#    validation_data=ds_validation
#)


modelplot = model.fit(ds_train, epochs=10 ,validation_data=ds_validation, batch_size = 64)

loss = modelplot.history['loss']
val_loss = modelplot.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs,loss,'y',label='Training loss')
plt.plot(epochs,val_loss,'r',label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = modelplot.history['acc']
val_acc = modelplot.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()