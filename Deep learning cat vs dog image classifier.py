#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import makedirs # to create working directory
from os import listdir # to list out working directory
from shutil import copyfile # copy file from source to destination to source
from random import seed,random


# In[2]:


from zipfile import ZipFile
# open zip file in system
with ZipFile("C:/Users/ankit/OneDrive/Desktop/datasets/dogs-vs-cats.zip",'r') as zipObj:
    zipObj.extractall()
with ZipFile("train.zip",'r') as zipObj:
    zipObj.extractall()
with ZipFile("test1.zip",'r') as zipObj:
    zipObj.extractall()


# In[5]:


# create directories
dataset_home = 'C:/Users/ankit/OneDrive/Desktop/datasets/dataset_dog_vs_cat/' 
subdirts= ['train/',
           'test/']
for subdir in subdirts:
    labeldirs=['dogs/',
               'cats/']
    for labeldir in labeldirs: 
        newdir= dataset_home+subdir+labeldir
        makedirs(newdir,exist_ok=True)
        


# In[7]:


seed(1)
val_ratio = 0.25
src_directory= 'train'
dataset_hom = 'C:/Users/ankit/OneDrive/Desktop/datasets/dataset_dog_vs_cat/'
for file in listdir(src_directory):
    src= src_directory+'/'+file
    dst_dir='train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('cat'):
        dst= dataset_hom+dst_dir+'cats/'+file
        copyfile(src,dst)
    elif file.startswith('dog'):
        dst= dataset_hom+dst_dir+'dogs/'+file
        copyfile(src,dst)


# In[9]:


import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


# In[16]:


model= VGG16(include_top=False,input_shape=(224,224,3))

#loaded layerd as not trainable
for layer in model.layers:
    layer.trainable = False

# add new layers which train classifiers

flat1=Flatten()(model.layers[-1].output)
class1=Dense(128,activation='relu',kernel_initializer='he_uniform')(flat1)
output=Dense(1,activation='sigmoid')(class1)

# define model
model= Model(inputs=model.inputs,outputs=output)
# compile

opt= SGD(learning_rate=0.001,momentum=0.9)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])


# In[17]:


model.summary()


# In[20]:


datagen = ImageDataGenerator(featurewise_center=True)

# define imagenet mean value for centering

datagen.mean=[123.68,116.779,103.939]

# prepare iterator

train_it= datagen.flow_from_directory('C:/Users/ankit/OneDrive/Desktop/datasets/dataset_dog_vs_cat/train',
                                     class_mode='binary',batch_size=64,target_size=(224,224))
test_it= datagen.flow_from_directory('C:/Users/ankit/OneDrive/Desktop/datasets/dataset_dog_vs_cat/test',
                                    class_mode='binary',batch_size=64,target_size=(224,224))
#fit model

history=model.fit_generator(train_it,steps_per_epoch=len(train_it),validation_data=test_it,
                           validation_steps=len(test_it),epochs=5,verbose=1)


# In[21]:


model.history.history.keys()


# In[64]:


import sys
import matplotlib.pyplot as plt
plt.subplot(211)
plt.title('Cross entropy loss')
plt.plot(history.history['loss'],color='b',label='train')
plt.plot(history.history['val_loss'],color='r',label='test')
plt.show()
plt.subplot(212)
plt.title('model accuracy')
plt.plot(history.history['accuracy'],color='b',label='train')
plt.plot(history.history['val_accuracy'],color='r',label='test')
plt.legend(loc='best')
# save plot to file
#filename= sys.argv[0].split('/')[-1]
#.savefig(filename+'_plot+png')


# In[27]:


model.save('cat_vs_dog.h5')


# In[39]:


from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model


# In[61]:


model=load_model('cat_vs_dog.h5')
img=load_img("C:/Users/ankit/OneDrive/Desktop/datasets/dataset_dog_vs_cat/test/cats/cat.620.jpg",target_size=(224,224))

#convert to array
img= img_to_array(img)

# reshape to single 3 channel image
img=img.reshape(1,224,224,3)

# center the pixel data
img= img.astype('float32')
img=img-[123.68,116.779,103.939]


# In[62]:


result=model.predict(img)
print(result[0])


# In[ ]:




