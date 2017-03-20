import os
import tensorflow as tf
import cv2
from keras.models import Sequential, model_from_json
from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

#Import path
train_dir = '/home/alvin/Documents/data-scientist/Pilot TJ/ROI training data/Gatsu-Pluit/Non-rect ROI/Train/'
test_dir = '/home/alvin/Documents/data-scientist/Pilot TJ/ROI training data/Gatsu-Pluit/Non-rect ROI/Test/'
model_dir = '/home/alvin/Documents/data-scientist/Pilot TJ/Estimation Model/Gatsu-Pluit/cnn/'

weight_model_filename = 'best_model_weights.hdf5'
architecture_model_filename = 'model_architecture.json'

#Set image data parameter (Changeable)
img_height,img_width = 288,352 #Image Dimension
resize_ratio = 0.35 #Image resize ratio
cnn_data_input_type = 'rgb' #Choose 'rgb' or 'gray'

#Set CNN Parameter (Convolutional Layer)
kernel_size = 3
pool_size = (2, 2)
strides = (2, 2)

#Set CNN Parameter (Optimizer)
batch_size = 10
nb_epoch = 500
optimizer = 'adam'
loss = 'mean_squared_error'


if cnn_data_input_type == 'rgb':
    depth=3
elif cnn_data_input_type == 'gray':
    depth=1
else:
    print("Error: Wrong cnn_data_input_type")

#Functions
def extract_data_from_images(images_path):
    image_list=os.listdir(images_path)
    image_data = np.zeros(((len(image_list)), 
                                 int(img_height*resize_ratio), 
                                 int(img_width*resize_ratio), 
                                 depth))
    image_gtlabel = []
    
    for counter in range(0,len(image_list)):
        image = cv2.imread(images_path+image_list[counter])
        image_gtlabel = np.append(image_gtlabel,
                                  int(float(image_list[counter].split(';')[0])))
        resized_image = cv2.resize(image,
                                   (int(img_width*resize_ratio),
                                    int(img_height*resize_ratio)))
        gray = cv2.cvtColor(resized_image,
                            cv2.COLOR_BGR2GRAY)
        if cnn_data_input_type == 'rgb':
            image_data[counter]=resized_image
        elif cnn_data_input_type == 'gray':
            gray = gray.reshape(gray.shape[0],
                                gray.shape[1],
                                depth)
            image_data[counter]=gray
    image_data = image_data.astype('float32')
    image_data /= 255
    
    #Reshape matrix dimension based on backend "theano" or "tensorflow"
    if K.image_dim_ordering() == 'th':
        image_data = image_data.reshape(image_data.shape[0],
                                        image_data.shape[3],
                                        image_data.shape[1],
                                        image_data.shape[2])
    else:
        image_data = image_data.reshape(image_data.shape[0],
                                        image_data.shape[1],
                                        image_data.shape[2],
                                        image_data.shape[3])
    return image_data,image_gtlabel

def root_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))**0.5

def make_model(train_data):
    print('Building CNN architecture model..')
    model = Sequential()
    
    model.add(Convolution2D(32,
                            kernel_size,
                            kernel_size,
                            border_mode='same',
                            input_shape=train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,
                            kernel_size,
                            kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size,
                           strides=strides))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64,
                            kernel_size,
                            kernel_size,
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,
                            kernel_size,
                            kernel_size))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size,
                           strides=strides))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[root_mean_squared_error])
    
    print('Finished building CNN architecture..')
    return model

def train_model(model,
                train_data,
                train_label,
                test_data,
                test_label,
                nb_epoch=100):

    checkpointer = ModelCheckpoint(filepath=model_dir+weight_model_filename,
                                   verbose=1,
                                   save_best_only=True)

    cnn_json_model = model.to_json()
    with open(model_dir+architecture_model_filename, "w") as json_file:
        json_file.write(cnn_json_model)
    
    print("Saved CNN architecture to disk..")
        
    print('Start optimizing CNN model..')
    model.fit(train_data,
              train_label,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(test_data, test_label),
              callbacks=[checkpointer],
              shuffle=True,
              verbose=1)
    
    print('Optimization finished..')
    return model


#Main Program
cnn_train_image_data,gtlabel_train_image_data=extract_data_from_images(train_dir)
cnn_test_image_data,gtlabel_test_image_data=extract_data_from_images(test_dir)

cnn_model = make_model(cnn_train_image_data)
trained_cnn_model = train_model(cnn_model,
                                cnn_train_image_data,
                                gtlabel_train_image_data,
                                cnn_test_image_data,
                                gtlabel_test_image_data,
                                nb_epoch)