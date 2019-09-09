
###############################
#Created by Kapil Paniker
#7 SEP 2019
###############################

###Importing Libraries#########
import sys
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import PIL
from IPython.display import display
from PIL import Image
from sklearn.preprocessing import LabelEncoder

#######################################
# Fixed parameters 
train_data_dir = 'D:/CNN/dogs-vs-cats/train'
validation_data_dir = 'D:/CNN/dogs-vs-cats/validation'
test_data_dir = 'D:/CNN/dogs-vs-cats/test'
nb_train_samples = 22000
nb_validation_samples = 3000
img_width = 224
img_height = 224
batch_size = 32

#######################################



#######################################
# Identify parameter format
#######################################
from keras import backend as K
if K.image_data_format() == 'channels_first':
    input_shp = (3, img_width, img_height)
else:
    input_shp = (img_width, img_height, 3)
#######################################



#Model layers definition

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shp))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(img_height, img_width),  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

test_datagen = ImageDataGenerator(rescale=1./255)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')
# Define the test generator
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')


model.fit_generator(
        train_generator,
        steps_per_epoch=688 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=70 // batch_size)
model.save_weights('D:/CNN/dogs-vs-cats/first_try.h5')
##################################################################################################################

##################################################################################################################
#Prebuilt Model

from keras.applications.vgg16 import VGG16
#model = VGG16()
#print(model.summary())
## Load the VGG16 Model ##########################################################################################
#vgg16_model = VGG16()
#vgg16_model = VGG16(weights="imagenet", include_top="false", input_shape=(224,224,3))
#vgg16_model.summary()
#type(vgg16_model)


#create a new sequential model and copy all the layers of VGG16 model except the last layer which is an output layer. 
#We have done this because we want our custom output layer which will have only two nodes as our 
#image classification problem has only two classes (cats and dogs).
#model_vgg = Sequential()
#for layer in vgg16_model.layers[:-1]:
#    model_vgg.add(layer)
# Freeze these layers 
#for layer in model_vgg.layers:
#    layer.trainable = False
#model_vgg.summary()
#Add 1 output dense layer for binary classification
#rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#model_vgg.add(Dense(128))
#model_vgg.add(Activation('relu'))
#model_vgg.add(Dense(1))
#model_vgg.add(Activation('sigmoid'))
#model_vgg.compile(loss='binary_crossentropy',
#              optimizer=rms,
#              metrics=['accuracy'])
#
#model_vgg.fit_generator(
#        train_generator,
#        steps_per_epoch=688 // batch_size,
#        epochs=50,
#        validation_data=validation_generator,
#        validation_steps=94 // batch_size)


###############################################################################################
#        Approach2
###############################################################################################
model_vgg = VGG16(include_top=False, input_shape=input_shp)
	# mark loaded layers as not trainable
for layer in model_vgg.layers:
    layer.trainable = False

flat1 = Flatten()(model_vgg.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
model = Model(inputs=model_vgg.inputs, outputs=output)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=688 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=94 // batch_size)

#Save Model Weights, Architecture, Optimizer conf
from keras.models import load_model
model.save('D:/CNN/dogs-vs-cats/my_model.h5')  

#Save Model Weights, Architecture, Optimizer conf - Full model load
#model = load_model('D:/CNN/dogs-vs-cats/my_model.h5')

# Save Weights
#model.save_weights('D:/CNN/dogs-vs-cats/pretrained_try.h5')

##Save Architecture
# save as JSON
#json_string = model.to_json()

#Load saved Json architecture
#from keras.models import model_from_json
#model = model_from_json(json_string)

# save as YAML
#yaml_string = model.to_yaml()

#Load saved YAML architecture
#from keras.models import model_from_yaml
#model = model_from_yaml(yaml_string)
##########################################################################################################
###############      Model Test       ####################################################################

from keras.models import load_model
from keras.models import model_from_json

#model = model_from_json(json_string)

#Load full model 
model = load_model('D:/CNN/dogs-vs-cats/my_model.h5')

#define the test generator transformations
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=1,
        shuffle=False,#this is important to map with filenames
        class_mode='binary')

#Predict on test set
filenames = test_generator.filenames

nb_test_samples = len(filenames)
#Get prediction probabilities for each image - 50 in this case
predict = model.predict_generator(test_generator,steps = nb_test_samples)
#Get accuracy of the classification
scores = model.evaluate_generator(test_generator,steps = nb_test_samples) #50 testing images
print("Accuracy = ", scores[1])

#combine the filenames and predictions to a dataframe for analysis

import numpy as np
output_data = pd.DataFrame(np.column_stack([filenames, predict]), 
                               columns=['Filename', 'Probability'])
output_data.head()

##############################################################################################################
##############################################################################################################
