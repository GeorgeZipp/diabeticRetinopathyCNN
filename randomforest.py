import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical, normalize
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import set_image_data_format
from keras.layers import BatchNormalization
import pickle


SIZE = 26

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))


X = X/255

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.1)

y_test=to_categorical(y_test)
y_train = to_categorical(y_train)

X_train = X_train.reshape(-1,26,26,1)
X_test = X_test.reshape(-1,26,26,1)
#y_test = y_test.reshape(-1,3)

set_image_data_format('channels_last')
print('X_train')
print(X_train)
print('X_test')
print(X_test)
print('y_train')
print(y_train)
print('y_test')
print(y_test)

X_train=tf.expand_dims(X_train, axis=-1)

X_test=tf.expand_dims(X_test, axis=-1)


activation = 'sigmoid'

feature_extractor = Sequential()
feature_extractor.add(Conv2D(16, 3, activation = activation, padding = 'same', input_shape = (SIZE, SIZE, 1)))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(16, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())

feature_extractor.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
feature_extractor.add(BatchNormalization())
feature_extractor.add(MaxPooling2D())

feature_extractor.add(Flatten())

XRF = feature_extractor.predict(X_train)

from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state=42)

RF_model.fit(XRF, y_train)

X_test_feature = feature_extractor.predict(X_test)

prediction_RF = RF_model.predict(X_test_feature)
#from sklearn import metrics
#print("Accuracy:",metrics.accuracy_score(X_test,prediction_RF))
yRF = np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yRF,prediction_RF)
sns.heatmap(cm, annot=True)