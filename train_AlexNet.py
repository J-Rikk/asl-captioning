#library for navigating directories
import os
#library for transforming images to numpy array
import numpy as np
#library for machine learning
import tensorflow as tf
#library for plotting graphs
import matplotlib.pyplot as plt

#Workstation-specific codes to access GPU and CUDA
from numba import cuda 

device = cuda.get_current_device()
device.reset()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#set directory where train dataset is located
train_directory = ".../train_dataset"

#set number of samples to estimate error gradient
batch_size=32
#AlexNet requires 227x227 images as input
img_height=227
img_width=227

#splits dataset to train data by 80%
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_directory,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#splits dataset to validation data by 20%
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_directory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#retrieves names of classes 
class_names = train_ds.class_names
print("Class names:",class_names)
print("Total classes:",len(class_names))

#modules for machine learning
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

#Alexnet architecture has 5 Convolutional layers, 3 Max Pooling layers, 3 Dropout layers, and 3 Fully Connected (Dense) layers
model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(filters=96, input_shape=(227,227,3), kernel_size=(11,11),strides=(4,4), padding='valid', activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
  layers.BatchNormalization(),
  layers.Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
  layers.BatchNormalization(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.BatchNormalization(),
  layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
  layers.BatchNormalization(),
  layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
  layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'),
  layers.BatchNormalization(),
  layers.Flatten(),
  layers.Dense(8192, input_shape=(227*227*3,), activation='relu'),
  layers.Dropout(0.7),
  layers.BatchNormalization(),
  layers.Dense(8192, activation='relu'),
  layers.Dropout(0.7),
  layers.BatchNormalization(),
  layers.Dense(4096, activation='relu'),
  layers.Dropout(0.6),
  layers.BatchNormalization(),
  #last dense layer defines number of classes (34)
  layers.Dense(34,activation='softmax')
])

model.summary()
#set learning rate close to the area of local minima to prevent underfitting and overfitting
sgd = tf.keras.optimizers.SGD(lr=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#saves the best model only from epoch where validation accuracy is the highest
checkpoint = ModelCheckpoint("/home/dseroy/CapstoneProject/03_Algorithm/Checkpoint/my_best_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#trains the model to 200 epochs
hist = model.fit(train_ds, batch_size=32,validation_batch_size=32, validation_data=valid_ds, epochs=200, callbacks=callbacks_list)
#saves numpy array where accuracy, loss, validation accuracy, and validation loss from all epochs are stored
np.save('.../train_history.npy',hist.history)

#plots accuracy and validation accuracy graph from 200 epochs
plt.plot(history['accuracy'], label = 'accuracy')
plt.plot(history['val_accuracy'], label = 'val_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend()
plt.savefig('.../accuracy_AlexNet.png')

#plots loss and validation loss graph from 200 epochs
plt.plot(history['loss'], label = 'loss')
plt.plot(history['val_loss'], label = 'val_loss')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')
plt.legend()
plt.savefig('.../loss_AlexNet.png')

#save model into h5 format, set directory where model will be saved
model.save(".../alexnet_model.h5")

#set directory where test dataset is located
test_directory = ".../test_dataset"

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow.keras.preprocessing.image as image

#stores actual label of test dataset
actual=[]
#stores predicted label of test dataset
pred=[]
#stores unique classes from test dataset
unique_list=[]

#traverses to test directory from folders to files
for root, subdirectories, files in os.walk(test_directory):
      for file in files:
        label = ""
        image_count = ""
        for char in file:
            #extracts image number from file name
            if char.isdigit():
                image_count += char
            #disregards underscores ( _ ) on label
            elif char == "_":
                continue
            #stops extraction when period (.) is encounter
            elif char == ".":
                break
            #extracts label from file name
            else:
                label += char

        #prints unique label once
        if label not in unique_list:
          print(label)
          unique_list.append(label)

        #appends actual label to the actual list
        actual.append(label)
        #loads the current image and turns it into a numpy array
        test_image = image.load_img(test_directory + "/" + label + "/" + file, target_size = (227, 227))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        #model predicts the class of current image and appends predicted label to prediction list
        result = model.predict(test_image)
        pred.append(class_names[np.argmax(result)])

#library for model evaluation
from sklearn.metrics import accuracy_score, classification_report

#prints test accuracy
print("Test accuracy=",accuracy_score(pred,actual))
#prints individual and overall reports on precision, recall, and F1 score up to four decimal values
print("Classification report:\n",classification_report(pred,actual, digits=4))

#library for constructing confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

#creates confusion matrix
confusion_mtx = confusion_matrix(actual, pred) 
confusion_mtx_percent = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]

#set dimensions of confusion matrix
fig, ax = plt.subplots(figsize=(20,12))

#set color to Greens, and percentages to whole numbers
sns.heatmap(confusion_mtx_percent, annot=True, fmt="0.0%", cmap='Greens');
ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); 
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)

#set rotation to fit phrases on x and y labels
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)

#saves confusion matrix into a png file
fig.savefig('.../matrix_Alexnet.png')