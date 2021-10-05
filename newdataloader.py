import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


DDIR = "C:/Users/alien/Desktop/CS Projects/diabetesResearchProject/patches/subpatches"
CATEGORIES = ["hard","normal","soft","hemorrhage"]

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DDIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (26,26),1)
            training_data.append([img_array, class_num])
create_training_data()
print(len(training_data))

import random
random.shuffle(training_data)

X = []
y = []
for features,label in training_data:
    X.append(features),
    y.append(label)



X = np.array(X)
y = np.array(y)
X.reshape((-1,26,26,1))

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()