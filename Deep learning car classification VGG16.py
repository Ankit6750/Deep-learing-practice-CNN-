#!/usr/bin/env python
# coding: utf-8

# In[2]:


from os import makedirs # to create working directory
from os import listdir # to list out working directory
from shutil import copyfile # copy file from source to destination to source
from random import seed,random


# In[3]:


from zipfile import ZipFile
# open zip file in system
with ZipFile("C:/Users/ankit/OneDrive/Desktop/datasets/car-dataset.zip",'r') as zipObj:
    zipObj.extractall('C:/Users/ankit/OneDrive/Desktop/datasets/dataset_cars/')


# In[41]:


import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from glob import glob


# In[42]:


# resize the image
image_size=[224,224]
train_path='C:/Users/ankit/OneDrive/Desktop/datasets/dataset_cars/Datasets/Train'
test_path='C:/Users/ankit/OneDrive/Desktop/datasets/dataset_cars/Datasets/Test'


# In[84]:


model= VGG16(include_top=False,input_shape=image_size+[3],weights='imagenet')

#loaded layerd as not trainable
for layer in model.layers:
    layer.trainable = False

    
folder=glob('C:/Users/ankit/OneDrive/Desktop/datasets/dataset_cars/Datasets/Train/*')
# add new layers which train classifiers
#x=Flatten()(model.output)
#prediction=Dense(len(folder),activation='softmax')(x)

flat1=Flatten()(model.layers[-1].output)
class1=Dense(100,activation='relu',kernel_initializer='glorot_uniform')(flat1)
output=Dense(len(folder),activation='softmax')(class1)

#create model
model=Model(inputs=model.input,outputs=output)


# In[85]:


model.summary()


# In[86]:


#opt=SGD(learning_rate=0.0001,momentum=0.9)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[96]:


train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('C:/Users/ankit/OneDrive/Desktop/datasets/dataset_cars/Datasets/Train',
                                     target_size=(224,224),
                                     batch_size=32,
                                     class_mode='categorical')
test_set = test_datagen.flow_from_directory('C:/Users/ankit/OneDrive/Desktop/datasets/dataset_cars/Datasets/Test',
                                    target_size=(224,224),
                                    batch_size=32,
                                    class_mode='categorical')
r= model.fit_generator(train_set,validation_data=test_set,epochs=5,steps_per_epoch=len(train_set),
                      validation_steps=len(test_set))


# In[97]:


r.history.keys()


# In[98]:


import matplotlib.pyplot as plt
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


# In[100]:


plt.plot(r.history['loss'], label='train acc')
plt.plot(r.history['val_loss'], label='val acc')
plt.legend()
plt.show()


# In[101]:


model.save('deep_cars.h5')


# In[102]:


from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model


# In[111]:


model=load_model('deep_cars.h5')
img=load_img("C:/Users/ankit/OneDrive/Desktop/datasets/dataset_cars/Datasets/Test/lamborghini/21.jpg",target_size=(224,224))

#convert to array
img= img_to_array(img)

# reshape to single 3 channel image
img=img.reshape(1,224,224,3)

# center the pixel data
img= img.astype('float32')
img=img-[123.68,116.779,103.939]


# In[112]:


result=model.predict(img)
print(result[0])


# In[ ]:




