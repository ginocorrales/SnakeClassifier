import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import keras
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

non_venomous_dir = "./Non-Venomous"
venomous_dir = "./Venomous"
snake_data = {}
label_dict = {}
target_size = (224,224,3)
venomous_species = []
non_venomous_species = []
#counter for species id
i = 0

for species in os.listdir(non_venomous_dir):
    non_venomous_species.append(species)
    img_list = []
    species_folder = os.listdir(os.path.join(non_venomous_dir,species))
    #print(species_folder)
    label_dict[species] = i
    for img_name in species_folder:
        img = load_img(non_venomous_dir + "/" + species + "/" + img_name)
        img = img.resize((224,224))
        img_array = img_to_array(img)
        img_list.append(img_array)
        
    snake_data[species] = np.array(img_list)
    i += 1
    
    
    
for species in os.listdir(venomous_dir):
    venomous_species.append(species)
    img_list = []
    species_folder = os.listdir(os.path.join(venomous_dir,species))
    label_dict[species] = i
    for img_name in species_folder:
        img = load_img(venomous_dir + "/" + species + "/" + img_name)
        img = img.resize((224,224))
        img_array = img_to_array(img)
        img_list.append(img_array)
        
    snake_data[species] = np.array(img_list)
    i += 1

img_array = []
img_labels = []
for species in snake_data:
    for img in snake_data[species]:
        img_array.append(img)
        img_labels.append(label_dict[species])
img_array = np.array(img_array)
img_labels = np.array(img_labels)

img_array,img_labels = shuffle(img_array,img_labels)

img_array[:][:][:][0] -= np.mean(img_array[:][:][:][0], axis = 0)
img_array[:][:][:][1] -= np.mean(img_array[:][:][:][1], axis = 0)
img_array[:][:][:][2] -= np.mean(img_array[:][:][:][2], axis = 0)

img_array[:][:][:][0] -= np.std(img_array[:][:][:][0], axis = 0)
img_array[:][:][:][1] -= np.std(img_array[:][:][:][1], axis = 0)
img_array[:][:][:][2] -= np.std(img_array[:][:][:][2], axis = 0)

train_percent = .8
train_count = int(train_percent*len(img_array))
test_count = len(img_array) - train_count

train_labels = img_labels[0:train_count]
test_labels = img_labels[train_count:]

train_images = img_array[0:train_count]
test_images = img_array[train_count:]

num_category = len(snake_data.keys())

# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_category)
test_labels = keras.utils.to_categorical(test_labels, num_category)


#Instantiate an empty model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# Passing it to a Fully Connected layer
model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# 3rd Fully Connected Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(num_category))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=["accuracy"])


#conv_model.fit(train_images,train_labels,epochs=5)
batch_size = 32
num_epoch = 30
#model training
model_log = model.fit(train_images,train_labels,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(test_images,test_labels))