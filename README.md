# an-MRI-based-deep-learning-approach-for-accurate-detection-of-Alzheimer-s-disease
# imports
import cv2
import numpy as np
import os
import cv2
import scipy.io

###############################################################################
# Define Variables
files = ['MildDemented',
         'ModerateDemented',
         'NonDemented',
         'VeryMildDemented'];

k = 0

# Creat Tensors
Data = np.arange(7216*100*100*1)
Data = np.reshape(Data, (7216,100,100))

t = []

# Load Data
for i in range(0,4): # i = 0, 1, 2, 3
    res = os.listdir('./MRI_Alzheimer/' + files[i])
    print(files[i] + ' : ' + str(len(res)) + ' images')

    for j in range(0,len(res)):
        img = cv2.imread('./MRI_Alzheimer/' + files[i] + '/' + res[j])
        img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize Data
        img_ = cv2.resize(img_,(100,100))
       # -*- coding: utf-8 -*-
"""
Deep learing model in keras
Implementation steps:
    1- Load data
    2- Creating layers and model
    3- Setting training parameters (Loss & optimization functions ,...)
    4- Training the model
    5- Network Evaluation
    6- Show Result
"""

import numpy as np
import scipy.io as sio
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import to_categorical
import seaborn as sns

## load data ##################################################################
data = sio.loadmat('Data.mat')
data = data['Data']
label = sio.loadmat('label.mat')
label = label['label']
label = list(label[0])

# Train and Test Sepratation
X_tr, X_ts, T_tr, T_ts = train_test_split(data, label ,
                                   random_state=104,
                                   test_size=0.25,
                                   shuffle=True)

T_tr = to_categorical(T_tr)
T_ts = to_categorical(T_ts)

# Loading Models ##################################################################
# InceptionV3
base_model = InceptionV3(weights=None, include_top=False, input_shape=(100,100,1))
base_model.trainable=True

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20,activation='relu')
dense_out = layers.Dense(4,activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_out
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', mode='max', patience=10)

model.summary()

# Train Network
history = model.fit(X_tr, T_tr, epochs = 100, batch_size=128, validation_split=0.2, callbacks=[es])

## network result #############################################################

Y_tr = model.predict(X_tr)
Y_tr1 = np.argmax(Y_tr, axis = 1)
Y_ts = model.predict(X_ts)
Y_ts1 = np.argmax(Y_ts, axis = 1)

T_tr1 = np.argmax(T_tr, axis = 1)
T_ts1 = np.argmax(T_ts, axis = 1)

## Calculate Result ###########################################################
cm_tr = confusion_matrix(T_tr1, Y_tr1)
acc_tr= accuracy_score(T_tr1, Y_tr1)*100
sen_tr = sum(recall_score(T_tr1, Y_tr1, average=None))/4*100
spe_tr = sum(precision_score(T_tr1, Y_tr1, average=None))/4*100

cm_ts = confusion_matrix(T_ts1, Y_ts1)
acc_ts= accuracy_score(T_ts1, Y_ts1)*100
sen_ts = sum(recall_score(T_ts1, Y_ts1, average=None))/4*100
spe_ts = sum(precision_score(T_ts1, Y_ts1, average=None))/4*100

# print result
print('Train Result :')
print("Accuracy    : ", f"{acc_tr:.2f}" , "%")
print("Sensitivity : ", f"{sen_tr:.2f}" , "%")
print("Specificity : ", f"{spe_tr:.2f}" , "%")
print('==================================')
print('Test Result :')
print("Accuracy    : ", f"{acc_ts:.2f}" , "%")
print("Sensitivity : ", f"{sen_ts:.2f}" , "%")
print("Specificity : ", f"{spe_ts:.2f}" , "%")
print('==================================')

## plot Result ################################################################
# plot Accuracy and Loss Function
# summarize history for accuracy
fig1 = plt.figure(figsize=(20, 10))
fig1.add_subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
fig1.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.savefig("./result/InceptionV3/training progress.png")

# Plot confusion matrix
fig2 = plt.figure(figsize=(20, 10))
fig2.add_subplot(1, 2, 1)
sns.heatmap(cm_tr, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

fig2.add_subplot(1, 2, 2)
sns.heatmap(cm_ts, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#plt.savefig("./result/InceptionV3/confusionmatrix.png")
# -*- coding: utf-8 -*-
"""
Deep learing model in keras
Implementation steps:
    1- Load data
    2- Creating layers and model
    3- Setting training parameters (Loss & optimization functions ,...)
    4- Training the model
    5- Network Evaluation
    6- Show Result
"""

import numpy as np
import scipy.io as sio
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import to_categorical
import seaborn as sns

## load data ##################################################################
data = sio.loadmat('Data.mat')
data = data['Data']
label = sio.loadmat('label.mat')
label = label['label']
label = list(label[0])

# Train and Test Sepratation
X_tr, X_ts, T_tr, T_ts = train_test_split(data, label ,
                                   random_state=104,
                                   test_size=0.25,
                                   shuffle=True)

T_tr = to_categorical(T_tr)
T_ts = to_categorical(T_ts)

# Loading Models ##################################################################
# DenseNet121
base_model = DenseNet121(weights=None, include_top=False, input_shape=(100,100,1))
base_model.trainable=True

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20,activation='relu')
dense_out = layers.Dense(4,activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_out
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', mode='max', patience=100)

model.summary()

# Train Network
history = model.fit(X_tr, T_tr, epochs = 100, batch_size=128, validation_split=0.2, callbacks=[es])

## network result #############################################################

Y_tr = model.predict(X_tr)
Y_tr1 = np.argmax(Y_tr, axis = 1)
Y_ts = model.predict(X_ts)
Y_ts1 = np.argmax(Y_ts, axis = 1)

T_tr1 = np.argmax(T_tr, axis = 1)
T_ts1 = np.argmax(T_ts, axis = 1)

## Calculate Result ###########################################################
cm_tr = confusion_matrix(T_tr1, Y_tr1)
acc_tr= accuracy_score(T_tr1, Y_tr1)*100
sen_tr = sum(recall_score(T_tr1, Y_tr1, average=None))/4*100
spe_tr = sum(precision_score(T_tr1, Y_tr1, average=None))/4*100

cm_ts = confusion_matrix(T_ts1, Y_ts1)
acc_ts= accuracy_score(T_ts1, Y_ts1)*100
sen_ts = sum(recall_score(T_ts1, Y_ts1, average=None))/4*100
spe_ts = sum(precision_score(T_ts1, Y_ts1, average=None))/4*100

# print result
print('Train Result :')
print("Accuracy    : ", f"{acc_tr:.2f}" , "%")
print("Sensitivity : ", f"{sen_tr:.2f}" , "%")
print("Specificity : ", f"{spe_tr:.2f}" , "%")
print('==================================')
print('Test Result :')
print("Accuracy    : ", f"{acc_ts:.2f}" , "%")
print("Sensitivity : ", f"{sen_ts:.2f}" , "%")
print("Specificity : ", f"{spe_ts:.2f}" , "%")
print('==================================')

## plot Result ################################################################
# plot Accuracy and Loss Function
# summarize history for accuracy
fig1 = plt.figure(figsize=(20, 10))
fig1.add_subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
fig1.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.savefig("./result/DenseNet121/training progress.png")

# Plot confusion matrix
fig2 = plt.figure(figsize=(20, 10))
fig2.add_subplot(1, 2, 1)
sns.heatmap(cm_tr, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

fig2.add_subplot(1, 2, 2)
sns.heatmap(cm_ts, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#plt.savefig("./result/DenseNet121/confusionmatrix.png")
# -*- coding: utf-8 -*-
"""
Deep learing model in keras
Implementation steps:
    1- Load data
    2- Creating layers and model
    3- Setting training parameters (Loss & optimization functions ,...)
    4- Training the model
    5- Network Evaluation
    6- Show Result
"""

import numpy as np
import scipy.io as sio
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.utils import to_categorical
import seaborn as sns

## load data ##################################################################
data = sio.loadmat('Data.mat')
data = data['Data']
label = sio.loadmat('label.mat')
label = label['label']
label = list(label[0])

# Train and Test Sepratation
X_tr, X_ts, T_tr, T_ts = train_test_split(data, label ,
                                   random_state=104,
                                   test_size=0.25,
                                   shuffle=True)

T_tr = to_categorical(T_tr)
T_ts = to_categorical(T_ts)

# Loading Models ##################################################################
# EfficientNetB7
base_model = EfficientNetB7(weights=None, include_top=False, input_shape=(100,100,1))
base_model.trainable=True

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20,activation='relu')
dense_out = layers.Dense(4,activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_out
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', mode='max', patience=100)

model.summary()

# Train Network
history = model.fit(X_tr, T_tr, epochs = 100, batch_size=128, validation_split=0.2, callbacks=[es])

## network result #############################################################

Y_tr = model.predict(X_tr)
Y_tr1 = np.argmax(Y_tr, axis = 1)
Y_ts = model.predict(X_ts)
Y_ts1 = np.argmax(Y_ts, axis = 1)

T_tr1 = np.argmax(T_tr, axis = 1)
T_ts1 = np.argmax(T_ts, axis = 1)

## Calculate Result ###########################################################
cm_tr = confusion_matrix(T_tr1, Y_tr1)
acc_tr= accuracy_score(T_tr1, Y_tr1)*100
sen_tr = sum(recall_score(T_tr1, Y_tr1, average=None))/4*100
spe_tr = sum(precision_score(T_tr1, Y_tr1, average=None))/4*100

cm_ts = confusion_matrix(T_ts1, Y_ts1)
acc_ts= accuracy_score(T_ts1, Y_ts1)*100
sen_ts = sum(recall_score(T_ts1, Y_ts1, average=None))/4*100
spe_ts = sum(precision_score(T_ts1, Y_ts1, average=None))/4*100

# print result
print('Train Result :')
print("Accuracy    : ", f"{acc_tr:.2f}" , "%")
print("Sensitivity : ", f"{sen_tr:.2f}" , "%")
print("Specificity : ", f"{spe_tr:.2f}" , "%")
print('==================================')
print('Test Result :')
print("Accuracy    : ", f"{acc_ts:.2f}" , "%")
print("Sensitivity : ", f"{sen_ts:.2f}" , "%")
print("Specificity : ", f"{spe_ts:.2f}" , "%")
print('==================================')

## plot Result ################################################################
# plot Accuracy and Loss Function
# summarize history for accuracy
fig1 = plt.figure(figsize=(20, 10))
fig1.add_subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
fig1.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.savefig("./result/EfficientNetB7/training progress.png")

# Plot confusion matrix
fig2 = plt.figure(figsize=(20, 10))
fig2.add_subplot(1, 2, 1)
sns.heatmap(cm_tr, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

fig2.add_subplot(1, 2, 2)
sns.heatmap(cm_ts, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#plt.savefig("./result/EfficientNetB7/confusionmatrix.png")
# -*- coding: utf-8 -*-
"""
Deep learing model in keras
Implementation steps:
    1- Load data
    2- Creating layers and model
    3- Setting training parameters (Loss & optimization functions ,...)
    4- Training the model
    5- Network Evaluation
    6- Show Result
"""

import numpy as np
import scipy.io as sio
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
import seaborn as sns

## load data ##################################################################
data = sio.loadmat('Data.mat')
data = data['Data']
label = sio.loadmat('label.mat')
label = label['label']
label = list(label[0])

# Train and Test Sepratation
X_tr, X_ts, T_tr, T_ts = train_test_split(data, label ,
                                   random_state=104,
                                   test_size=0.25,
                                   shuffle=True)

T_tr = to_categorical(T_tr)
T_ts = to_categorical(T_ts)

# Loading Models ##################################################################
# ResNet50
base_model = ResNet50(weights=None, include_top=False, input_shape=(100,100,1))
base_model.trainable=True

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20,activation='relu')
dense_out = layers.Dense(4,activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_out
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', mode='max', patience=100)

model.summary()

# Train Network
history = model.fit(X_tr, T_tr, epochs = 100, batch_size=128, validation_split=0.2, callbacks=[es])

## network result #############################################################

Y_tr = model.predict(X_tr)
Y_tr1 = np.argmax(Y_tr, axis = 1)
Y_ts = model.predict(X_ts)
Y_ts1 = np.argmax(Y_ts, axis = 1)

T_tr1 = np.argmax(T_tr, axis = 1)
T_ts1 = np.argmax(T_ts, axis = 1)

## Calculate Result ###########################################################
cm_tr = confusion_matrix(T_tr1, Y_tr1)
acc_tr= accuracy_score(T_tr1, Y_tr1)*100
sen_tr = sum(recall_score(T_tr1, Y_tr1, average=None))/4*100
spe_tr = sum(precision_score(T_tr1, Y_tr1, average=None))/4*100

cm_ts = confusion_matrix(T_ts1, Y_ts1)
acc_ts= accuracy_score(T_ts1, Y_ts1)*100
sen_ts = sum(recall_score(T_ts1, Y_ts1, average=None))/4*100
spe_ts = sum(precision_score(T_ts1, Y_ts1, average=None))/4*100

# print result
print('Train Result :')
print("Accuracy    : ", f"{acc_tr:.2f}" , "%")
print("Sensitivity : ", f"{sen_tr:.2f}" , "%")
print("Specificity : ", f"{spe_tr:.2f}" , "%")
print('==================================')
print('Test Result :')
print("Accuracy    : ", f"{acc_ts:.2f}" , "%")
print("Sensitivity : ", f"{sen_ts:.2f}" , "%")
print("Specificity : ", f"{spe_ts:.2f}" , "%")
print('==================================')

## plot Result ################################################################
# plot Accuracy and Loss Function
# summarize history for accuracy
fig1 = plt.figure(figsize=(20, 10))
fig1.add_subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
fig1.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.savefig("./result/ResNet50/training progress.png")

# Plot confusion matrix
fig2 = plt.figure(figsize=(20, 10))
fig2.add_subplot(1, 2, 1)
sns.heatmap(cm_tr, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

fig2.add_subplot(1, 2, 2)
sns.heatmap(cm_ts, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#plt.savefig("./result/ResNet50/confusionmatrix.png")
# -*- coding: utf-8 -*-
"""
Deep learing model in keras
Implementation steps:
    1- Load data
    2- Creating layers and model
    3- Setting training parameters (Loss & optimization functions ,...)
    4- Training the model
    5- Network Evaluation
    6- Show Result
"""

import numpy as np
import scipy.io as sio
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
import seaborn as sns

## load data ##################################################################
data = sio.loadmat('Data.mat')
data = data['Data']
label = sio.loadmat('label.mat')
label = label['label']
label = list(label[0])

# Train and Test Sepratation
X_tr, X_ts, T_tr, T_ts = train_test_split(data, label ,
                                   random_state=104,
                                   test_size=0.25,
                                   shuffle=True)

T_tr = to_categorical(T_tr)
T_ts = to_categorical(T_ts)

# Loading Models ##################################################################
# VGG16
base_model = VGG16(weights=None, include_top=False, input_shape=(100,100,1))
base_model.trainable=True

flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(50, activation='relu')
dense_layer_2 = layers.Dense(20,activation='relu')
dense_out = layers.Dense(4,activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_out
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', mode='max', patience=100)

model.summary()

# Train Network
history = model.fit(X_tr, T_tr, epochs = 100, batch_size=128, validation_split=0.2, callbacks=[es])

## network result #############################################################

Y_tr = model.predict(X_tr)
Y_tr1 = np.argmax(Y_tr, axis = 1)
Y_ts = model.predict(X_ts)
Y_ts1 = np.argmax(Y_ts, axis = 1)

T_tr1 = np.argmax(T_tr, axis = 1)
T_ts1 = np.argmax(T_ts, axis = 1)

## Calculate Result ###########################################################
cm_tr = confusion_matrix(T_tr1, Y_tr1)
acc_tr= accuracy_score(T_tr1, Y_tr1)*100
sen_tr = sum(recall_score(T_tr1, Y_tr1, average=None))/4*100
spe_tr = sum(precision_score(T_tr1, Y_tr1, average=None))/4*100

cm_ts = confusion_matrix(T_ts1, Y_ts1)
acc_ts= accuracy_score(T_ts1, Y_ts1)*100
sen_ts = sum(recall_score(T_ts1, Y_ts1, average=None))/4*100
spe_ts = sum(precision_score(T_ts1, Y_ts1, average=None))/4*100

# print result
print('Train Result :')
print("Accuracy    : ", f"{acc_tr:.2f}" , "%")
print("Sensitivity : ", f"{sen_tr:.2f}" , "%")
print("Specificity : ", f"{spe_tr:.2f}" , "%")
print('==================================')
print('Test Result :')
print("Accuracy    : ", f"{acc_ts:.2f}" , "%")
print("Sensitivity : ", f"{sen_ts:.2f}" , "%")
print("Specificity : ", f"{spe_ts:.2f}" , "%")
print('==================================')

## plot Result ################################################################
# plot Accuracy and Loss Function
# summarize history for accuracy
fig1 = plt.figure(figsize=(20, 10))
fig1.add_subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
fig1.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.savefig("./result/vgg16/training progress.png")

# Plot confusion matrix
fig2 = plt.figure(figsize=(20, 10))
fig2.add_subplot(1, 2, 1)
sns.heatmap(cm_tr, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

fig2.add_subplot(1, 2, 2)
sns.heatmap(cm_ts, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#plt.savefig("./result/vgg16/confusionmatrix.png")
# -*- coding: utf-8 -*-
"""
Deep learing model in keras
Implementation steps:
    1- Load data
    2- Creating layers and model
    3- Setting training parameters (Loss & optimization functions ,...)
    4- Training the model
    5- Network Evaluation
    6- Show Result
"""

import numpy as np
import scipy.io as sio
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import seaborn as sns

## load data ##################################################################
data = sio.loadmat('Data.mat')
data = data['Data']
label = sio.loadmat('label.mat')
label = label['label']
label = list(label[0])

# Train and Test Sepratation
X_tr, X_ts, T_tr, T_ts = train_test_split(data, label ,
                                   random_state=104,
                                   test_size=0.25,
                                   shuffle=True)

T_tr = to_categorical(T_tr)
T_ts = to_categorical(T_ts)

# Loading Models ##################################################################
# Proposed Model
# Initialize the model
model = models.Sequential()

# Convolutional Block 1
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(100,100,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Conv2D(512, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
model.add(Dropout(0.3))

# Fully Connected Dense Layer
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(4, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

es = EarlyStopping(monitor='val_loss', mode='max', patience=100)

model.summary()

# Train Network
history = model.fit(X_tr, T_tr, epochs = 100, batch_size=128, validation_split=0.2, callbacks=[es])

## network result #############################################################

Y_tr = model.predict(X_tr)
Y_tr1 = np.argmax(Y_tr, axis = 1)
Y_ts = model.predict(X_ts)
Y_ts1 = np.argmax(Y_ts, axis = 1)

T_tr1 = np.argmax(T_tr, axis = 1)
T_ts1 = np.argmax(T_ts, axis = 1)

## Calculate Result ###########################################################
cm_tr = confusion_matrix(T_tr1, Y_tr1)
acc_tr= accuracy_score(T_tr1, Y_tr1)*100
sen_tr = sum(recall_score(T_tr1, Y_tr1, average=None))/4*100
spe_tr = sum(precision_score(T_tr1, Y_tr1, average=None))/4*100

cm_ts = confusion_matrix(T_ts1, Y_ts1)
acc_ts= accuracy_score(T_ts1, Y_ts1)*100
sen_ts = sum(recall_score(T_ts1, Y_ts1, average=None))/4*100
spe_ts = sum(precision_score(T_ts1, Y_ts1, average=None))/4*100

# print result
print('Train Result :')
print("Accuracy    : ", f"{acc_tr:.2f}" , "%")
print("Sensitivity : ", f"{sen_tr:.2f}" , "%")
print("Specificity : ", f"{spe_tr:.2f}" , "%")
print('==================================')
print('Test Result :')
print("Accuracy    : ", f"{acc_ts:.2f}" , "%")
print("Sensitivity : ", f"{sen_ts:.2f}" , "%")
print("Specificity : ", f"{spe_ts:.2f}" , "%")
print('==================================')

## plot Result ################################################################
# plot Accuracy and Loss Function
# summarize history for accuracy
fig1 = plt.figure(figsize=(20, 10))
fig1.add_subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
fig1.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

#plt.savefig("./result/Proposed_Model/training progress.png")

# Plot confusion matrix
fig2 = plt.figure(figsize=(20, 10))
fig2.add_subplot(1, 2, 1)
sns.heatmap(cm_tr, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

fig2.add_subplot(1, 2, 2)
sns.heatmap(cm_ts, annot=True, fmt="d", cmap="Blues", xticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'],
            yticklabels=['NonDemented','VeryMildDemented','MildDemented','ModerateDemented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#plt.savefig("./result/Proposed_Model/confusionmatrix.png")













 
