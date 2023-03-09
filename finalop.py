
# data visualisation and manipulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
 

#model selection

from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

#dl libraraies
import tensorflow as tf
import random as rn

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
from tqdm import tqdm
import os                   
from random import shuffle  


from tqdm import tqdm
X=[]
Z=[]
def load_data(document,DIR):
        for img in tqdm(os.listdir(DIR)):
            label = document
            path = os.path.join(DIR,img)
            image= load_img(path,target_size=(IMG_SIZE,IMG_SIZE))
            image= img_to_array(image)
            image = preprocess_input(image)

            X.append(image)
            Z.append(str(label))
        return X,Z
        IMG_SIZE=256


X=[]
Z=[]
DIR_email='C:/Users/noohh/OneDrive/Documents/image-classifcation/email'
DIR_resume='C:/Users/noohh/OneDrive/Documents/image-classifcation/resume'
DIR_publication='C:/Users/noohh/OneDrive/Documents/image-classifcation/scientific_publication'

load_data('email',DIR_email)
print(len(X))
load_data('resume',DIR_resume)
print(len(X))
load_data('scientific_publication',DIR_publication)
print(len(X))



plt.figure(figsize=(10,10))
#fig,ax = plt.subplots(5,5)

for i in range(25):
  plt.subplot(5,5,1+i)
  l = rn.randint(0,140)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(X[l])
  plt.xlabel(Z[l])
plt.show()




from sklearn.model_selection import train_test_split

le = LabelEncoder()
Y = le.fit_transform(Z)
Y = to_categorical(Y,3)
Y[150]


X = np.array(X)
X=X/255

print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8,random_state=42)



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




from tensorflow.keras.optimizers import Adam

# compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test,y_test))



def get_input(filename):
    import numpy as np

    # load the image and resize it to (224, 224) which is the input size for VGG16 model
    img = load_img(filename, target_size=(256, 256,3))

    # convert the image to a numpy array
    img_arr = img_to_array(img)

    # expand the dimensions of the image to match the input shape of the model
    img_arr = np.expand_dims(img_arr, axis=0)

    # normalize the pixel values of the image
    img_arr /= 255

    # pass the preprocessed image to the model for prediction
    predictions = list(model.predict(img_arr))
    op=list(predictions[0])
    max_op=max(op)
    index=op.index(max_op)
    if index==0:
        return('E-mail')
    elif index==1:
        return('Resume')
    else:
        return('Scientific Publications')
    



