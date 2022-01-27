#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__


# - Fashion minst dataset output rsult should be like
# - 0: T-shirt/top
# - 1: Trouser
# - 2: Pullover
# - 3: Dress
# - 4: Coat
# - 5: Sandal
# - 6: Shirt
# - 7: Sneaker
# - 8: Bag
# - 9: Ankle boot

# In[ ]:


#!pip install python3-utils


# In[40]:


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.datasets import fashion_mnist 
from keras import utils


# In[4]:


(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()


# In[11]:


y_train[0]


# In[20]:


y_train.shape


# In[21]:


x_train.shape


# In[18]:


fig,ax = plt.subplots(ncols=10,sharex=False,sharey=True,figsize=(20,5))
for i in range(10):
    ax[i].set_title(y_train[i])
    ax[i].imshow(x_train[i],cmap='gray')
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(True)
plt.show()    


# In[19]:


x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test= x_test.reshape(x_test.shape[0],28,28,1)


# In[23]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


# In[24]:


y_train.shape


# In[25]:


y_train[0]


# In[79]:


model=Sequential()
# add conv layer
model.add(Conv2D(filters=32,input_shape=(28,28,1),kernel_size=(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))

# add maxpooling
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add((Dense(50,activation='relu',kernel_initializer='he_uniform')))
model.add(Dense(10,activation='softmax'))


# In[80]:


model.summary()


# In[82]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=30,epochs=5,verbose=1,validation_data=(x_test,y_test))


# In[83]:


model.save('fashion_minst_cnn.h5')


# In[84]:


print(model.history.history.keys())


# In[85]:


print(model.history.history)


# In[86]:


plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','test'],loc='upper left')
plt.show()


# In[87]:


model=keras.models.load_model('fashion_minst_cnn.h5')


# In[93]:


plt.imshow(x_test[20],cmap='gray')


# In[94]:


x=np.reshape(x_test[20],(1,28,28,1))
np.argmax(model.predict(x))


# In[ ]:




