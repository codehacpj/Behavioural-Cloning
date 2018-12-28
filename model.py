##Loading data
import os
import csv


samples = [] 

with open('./data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None) #incase of heading
    for line in reader:
        samples.append(line)
        
#Train/Validation split for the data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


train_samples, validation_samples = train_test_split(samples,test_size=0.15)

import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt


def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: 
        shuffle(samples) #shuffle
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3): #images corresponding to each camera
                        
                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) #BGR to RGB
                        center_angle = float(batch_sample[3]) #steering angle
                        images.append(center_image)
                        
                        #Correction Factor 0.2
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)
                        
                        # Data Augumentation(flip)
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1) 
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) 

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D

model = Sequential()

# Preprocessing the incoming data (normalize)
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# cropping images
model.add(Cropping2D(cropping=((70,25),(0,0))))           

#layer 1- Convolution, number of filters- 24, filter size= 5x5, stride= 2x2
model.add(Conv2D(24, (5, 5), strides=(2, 2)))
model.add(Activation('elu'))

#layer 2- Convolution, number of filters- 36, filter size= 5x5, stride= 2x2
model.add(Conv2D(36, (5, 5), strides=(2, 2)))
model.add(Activation('elu'))

#layer 3- Convolution, number of filters- 48, filter size= 5x5, stride= 2x2
model.add(Conv2D(48, (5, 5), strides=(2, 2)))
model.add(Activation('elu'))

#layer 4- Convolution, number of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64, (3, 3)))
model.add(Activation('elu'))

#layer 5- Convolution, number of filters- 64, filter size= 3x3, stride= 1x1
model.add(Conv2D(64, (3, 3)))
model.add(Activation('elu'))

#flatten image from 2D to side by side
model.add(Flatten())

#layer 6- fully connected layer 1
model.add(Dense(100))
model.add(Activation('elu'))

#Adding a dropout layer to avoid overfitting.
model.add(Dropout(0.25))

#layer 7- fully connected layer 1
model.add(Dense(50))
model.add(Activation('elu'))


#layer 8- fully connected layer 1
model.add(Dense(10))
model.add(Activation('elu'))

#layer 9- fully connected layer 1
model.add(Dense(1)) #regression o/p


# the output is the steering angle
# using mean squared error loss function is the right choice for this regression problem
# adam optimizer 
model.compile(loss='mse',optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

#saving model
model.save('model.h5')

# keras method to print the model summary
model.summary()

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()