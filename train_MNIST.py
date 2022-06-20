#library for accessing CSV files
import pandas as pd
#library for transforming images to numpy array
import numpy as np
#library for plotting graphs
import matplotlib.pyplot as plt
#library for displaying images
import matplotlib.image as mpimg

#library for splitting dataset to train and validation data
from sklearn.model_selection import train_test_split

#library for machine learning
import tensorflow as tf
#modules for machine learning
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#loads CSV files for train and test datasets
train = pd.read_csv("C:/Users/Abdul-Aziz Tahil/Desktop/Still_Images/sign_mnist_train.csv")
test = pd.read_csv("C:/Users/Abdul-Aziz Tahil/Desktop/Still_Images/sign_mnist_test.csv")

#defines train and test datasets
X_train = train.drop(labels=['label'], axis=1)
Y_train = train['label']
X_test = test.drop(labels=['label'], axis=1)
Y_test = test['label']

#shows image from dataset
def gen_image(image):
    """Return 28x28 image given grayscale values"""
    pixels = image.reshape((28,28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

#assigns numbers to their corresponding label
label_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',\
             18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y'}
label_dict_rev = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'O':13,'P':14,'Q':15,'R':16,\
                 'S':17,'T':18,'U':19,'V':20,'W':21,'X':22,'Y':23}

Y_test1 = []
for i in Y_test:
    Y_test1.append(label_dict_rev.get(label_dict.get(i)))

Y_test2 = to_categorical(Y_test1, num_classes = 24)
Y_train = train['label']

Y_train1 = []
for i in Y_train:
    Y_train1.append(label_dict_rev.get(label_dict.get(i)))
    
#classes labeled as 1-24; Need to change to 0-23 for to_categorical
Y_train2 = to_categorical(Y_train1, num_classes = 24)
X_train = train.drop(labels=['label'], axis=1)

#normalizes pixels
X_train1 = X_train/255
X_test1 = X_test/255

#converts data to 2D form to represent height x width
X_train2 = X_train1.values.reshape(-1,28,28,1)
X_test2 = X_test1.values.reshape(-1,28,28,1)
#splits training set into 10% validation and 90% training
X_tr, X_val, Y_tr, Y_val = train_test_split(X_train2, Y_train2, test_size = 0.3, random_state=2, stratify=Y_train2)

#MNIST CNN architecture has 2 Convolutional layers, 2 Max Pooling layers, 3 Dropout layers and 2 Fully Connected layers
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2))) # downsampling
model.add(Dropout(0.25)) # Dropout reduces overfitting
model.add(Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
#last dense layer defines number of classes (24)
model.add(Dense(24, activation='softmax'))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

#uses categorical cross entropy as loss function
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#defines model parameters: number of epochs, batch size, and learning rate reduction
epochs = 10
batch_size = 64
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

model.summary()
#trains the model to 10 epochs, adjusting learning rate if validation accuracy does not improve
hist = model.fit(X_train2, Y_train2, batch_size=batch_size, epochs=epochs, validation_data=(X_val,Y_val), callbacks=[learning_rate_reduction])
model.evaluate(X_test2, Y_test2)

np.save('.../train_history.npy',hist.history)

#plots accuracy and validation accuracy graph from 10 epochs
plt.plot(history['accuracy'], label = 'accuracy')
plt.plot(history['val_accuracy'], label = 'val_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend()
plt.savefig('.../accuracy_MNIST.png')

#plots loss and validation loss graph from 10 epochs
plt.plot(history['loss'], label = 'loss')
plt.plot(history['val_loss'], label = 'val_loss')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')
plt.legend()
plt.savefig('.../loss_MNIST.png')

#save model into h5 format, set directory where model will be saved
model.save('.../mnist_model.h5')

results = model.predict(X_test2) # predict test labels
Y_pred_classes = np.argmax(results, axis = 1) # Convert predictions classes to one hot vectors 
Y_true = np.argmax(Y_test2,axis = 1) # Convert validation observations to one hot vectors

#library for model evaluation
from sklearn.metrics import accuracy_score, classification_report

#prints test accuracy
print("Test accuracy=",accuracy_score(Y_true, Y_pred_classes))
#prints individual and overall reports on precision, recall, and F1 score up to four decimal values
print("Classification report:\n",classification_report(Y_true, Y_pred_classes, digits=4))

#library for constructing confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
confusion_mtx_percent = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]

#set dimensions of confusion matrix
fig, ax = plt.subplots(figsize=(20,12))

#set color to Greens, and percentages to whole numbers
sns.heatmap(confusion_mtx_percent, annot=True, fmt="0.0%", cmap='Greens');
labels=['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels'); 
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

#saves confusion matrix into a png file
fig.savefig('.../matrix_MNIST.png')


