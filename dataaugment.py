from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image as im
from skimage import io

datagen = ImageDataGenerator(
    rotation_range=0.45
)

import numpy as np
import os
from PIL import Image
saved_dir = 'patches/augment/new'
image_directory = 'patches/augment/soft/'
SIZE = 26
dataset = []

my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if(image_name.split('.')[1] == 'png'):
        image = io.imread(image_directory + image_name)
        dataset.append(np.array(image))

x = np.array(dataset)
print(x.shape)
x = x[:,:,:,np.newaxis]
print(x.shape)
i=0
f=0
for batch in datagen.flow(x, batch_size = 1, save_to_dir='patches/augment/augmented',save_prefix='aug',save_format='png'):
    img_save = tf.keras.preprocessing.image.array_to_img(batch[0], scale=False)
    img_save.save('patches/augment/augmentedsoft' + fr'\augment_{f}.png')
    f += 1
    i += 1
    if i > 1000:
        break

#i = 0
#for batch in datagen.flow_from_directory(directory='patches/augment/',
#                          batch_size=16,
#                          target_size=(SIZE,SIZE),
#                          color_mode="grayscale",
#                          save_to_dir="hardaug",
#                          save_prefix="aug",
#                          save_format="png"
#                          ):
#    i = i+1
#    if i > 50:
#        break
