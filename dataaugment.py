from keras.preprocessing.image import ImageDataGenerator
from skimage import io

datagen = ImageDataGenerator(
    rotation_range=0.45,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode=('reflect')
)

import numpy as np
import os
from PIL import Image
saved_dir = ''
image_directory = 'patches/subpatches/normal'
SIZE = 26
dataset = []

#my_images = os.listdir(image_directory)
#for i, image_name in enumerate(my_images):
#    if(image_name.split('.')[1] == 'png'):
#        image = io.imread(image_directory + image_name)
#        image = Image.fromarray(image, 'RGB')
#        image = image.resize((SIZE,SIZE))
#        dataset.append(np.array(image))
x = io.imread('patches/subpatches/hard/')
i = 0
for batch in datagen.flow(x,
                          batch_size=16,
                          target_size=(SIZE,SIZE),
                          color_mode="rgb",
                          save_to_dir="hardaug",
                          save_prefix="aug",
                          save_format="png"
                          ):
    i = i+1