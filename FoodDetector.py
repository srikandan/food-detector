# Importing required packages
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

# Train Images Path
DATADIR = r"C:\Users\SK\FoodDetector\TrainImages"

# Food Categories
CATGOERIES = ["Dosa","Idiyappam","Idli","Onion","Pongal","Poori","Vada"]

imagesize = 50
training_dataset  = []

def create_training_dataset():
    """
    Converts images into array values

    Parameters
    ----------
    None.

    Returns
    -------
    None.

    """
    for Category in CATGOERIES:
        path = os.path.join(DATADIR,Category)
        class_num = CATGOERIES.index(Category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(imagesize,imagesize))
                training_dataset.append([new_array,class_num])
            except:
                pass
 
create_training_dataset()

# Shuffling the Dataset for better prediction
random.shuffle(training_dataset)

# Select X and Y
X=[]
Y=[]

for features,label in training_dataset:
    X.append(features)
    Y.append(label)
    
X = np.array(X)
Y = np.array(Y)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=101)

# Scaling - Normaizsation
X_train  = tf.keras.utils.normalize(X_train,axis=1)
X_test  = tf.keras.utils.normalize(X_test,axis=1)

# Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(8,activation=tf.nn.softmax))

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Model fitting
model.fit(X_train,y_train,epochs=30)

# Value accuracy of the model
val_loss, val_acc = model.evaluate(X_train,y_train)