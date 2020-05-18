import tensorflow as tf
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
import matplotlib.image as imp
import numpy as np
import csv


lines = []
with open('/home/workspace/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
for line in lines:
    source_path = line[0]
    image_path = source_path.split('/')[-1]
    current_path = '/opt/carnd_p3/data/IMG/' + image_path
    image = imp.imread(current_path)
    measurement = float(line[3])
    images.append(image)
    measurements.append(measurement)
    
    source_path = line[1]
    image_path = source_path.split('/')[-1]
    current_path = '/opt/carnd_p3/data/IMG/' + image_path
    image = imp.imread(current_path)
    flip_image = np.fliplr(image)
    images.append(image)
    images.append(flip_image)
    measurement = float(line[3])+0.2
    flip_measure = -measurement
    measurements.append(measurement)
    measurements.append(flip_measure)
    
    source_path = line[2]
    image_path = source_path.split('/')[-1]
    current_path = '/opt/carnd_p3/data/IMG/' + image_path
    image = imp.imread(current_path)
    flip_image = np.fliplr(image)
    images.append(image)
    images.append(flip_image)
    measurement = float(line[3])-0.2
    flip_measure = -measurement
    measurements.append(measurement)
    measurements.append(flip_measure)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
# Crop the images
model.add(Cropping2D(cropping=((50,25), (0,0)), input_shape=(160,320,3)))
# Normalize the images
model.add(Lambda(lambda x: x/255-0.5))

# Layer1: Convolutional-relu-Maxpool.
#Input=(85,320,3). Output=(41,106,6)
model.add(Conv2D(6, kernel_size=(5,5), strides=(2,3), padding='valid', activation='relu'))
#Input=(41,106,6). Output=(20,52,6)
model.add(MaxPooling2D(pool_size=(3,4), strides=(2,2), padding='valid'))

# Layer2: Convolutional-relu_Maxpool-Dropout. 
#Input=(20,52,6). Output=(18,50,10)
model.add(Conv2D(10, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
# Input=(18,50,10). Output=(17,49,10)
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
model.add(Dropout(0.2))

# Layer3: Convolutional-relu-Maxpool-Droput.
# Input=(17,49,10). Output(15,47,14)
model.add(Conv2D(14, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
# Input=(15,47,14). Output=(14,46,14)
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
model.add(Dropout(0.2))

# Layer4: Convolutional-relu-Maxpool-Dropout.
# Input=(14,46,14). Output=(12,44,16)
model.add(Conv2D(16, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'))
#Input=(12,44,16). Output=(11,43,16)
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
model.add(Dropout(0.2))

# Layer5: Convolutional-relu-Maxpool-Dropout.
# Input=(11,43,16). Output=(5,21,18)
model.add(Conv2D(18, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))
#Input=(5,21,18). Output=(4,20,18)
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))
model.add(Dropout(0.2))

#Flatten
model.add(Flatten())

# Layer5: Fully connected-relu-Dropout.
# Input=1440. Output=120
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.2))

# Layer6: Fully connected-relu-Dropout
# Input=120. Output=10.
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))

# Layer7: Fully connected-relu
# Input=10. Output=1
model.add(Dense(1))

# Train the model
batch_size=128
epoch = 10

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0.2, shuffle=True)
#plt.plot(history.history['loss'], label='training_loss')
#plt.plot(history.history['val_loss'], label='validation_loss')
#plt.title('Mean Squared Erro Loss')
#plt.ylabel('mse')
#plt.xlabel('epoch')
#plt.legend()
#plt.show()

model.save('model.h5')
