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

activation = 'relu'
model = Sequential()
model.add(Conv2D(16, 3, activation=activation,padding='same',input_shape=(26,26,1)))
model.add(BatchNormalization())
model.add(Conv2D(16, 3, activation=activation,padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,3,activation=activation,padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32,3,activation=activation,padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(32,activation=activation,kernel_initializer='he_uniform'))
model.add(Dense(4, activation='softmax'))

#print(model.summary())
model.compile(optimizer='rmsprop',loss="categorical_crossentropy",metrics=["accuracy"])

#train_datagen = ImageDataGenerator(
#    rescale=1./255,
#    rotation_range=45,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True
#)

#validation_datagen = ImageDataGenerator(rescale=1./255)

#train_generator = train_datagen.flow_from_directory(
#    'patches/subpatches',
#    color_mode='grayscale',
#    target_size =(26,26),
#    batch_size=16
#)

#validation_generator = validation_datagen.flow_from_directory(
#    'patches/subpatches',
#    color_mode='grayscale',
#    target_size = (26,26),
#    batch_size=16
#)

from keras.callbacks import ModelCheckpoint

filepath ="saved_models/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=32,callbacks=callbacks_list)

#history = model.fit(
#    train_generator,
#    steps_per_epoch=2000,
#    epochs=10,
#    validation_data=validation_generator,
#    validation_steps=800,
#)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

rounded_predictions = np.argmax(model.predict(X_test,batch_size=16, verbose=0),axis=-1)
rounded_labels = np.argmax(y_test,axis=-1)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(rounded_labels,rounded_predictions)
print(cm)

print(classification_report(rounded_labels, rounded_predictions, target_names=['hard', 'hemo', 'normal','soft']))


#Pred = model.predict(X_test, batch_size=32)
#Pred_Label = np.argmax(Pred, axis=1)

#test_acc = accuracy_score(y_test, rounded_labels)
#ConfusionM = confusion_matrix(list(y_test), rounded_labels, labels=[0, 1, 2, 3])
#class_report = classification_report(list(y_test), rounded_labels, labels=[0, 1, 2, 3])

#from sklearn.metrics import roc_curve
#y_pred_keras = model.predict(X_test).ravel()
#fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
