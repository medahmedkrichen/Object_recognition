from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print(x_train.shape)

for i in range(9):
  plt.subplot(330+1+i)
  img=x_train[i]
  plt.imshow(img)
  
np.random.seed(6)

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_test/=255
x_train/=255

Y_train=np_utils.to_categorical(y_train)
Y_test=np_utils.to_categorical(y_test)

from keras import Sequential
from keras.layers import Dropout,Activation,GlobalAveragePooling2D,Conv2D
from keras.optimizers import SGD


def allcnn(weights=None):
    
    model = Sequential()

    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding = 'valid'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    if weights:
        model.load_weights(weights)
    
    return model


learning_rate=0.01
weights_decay=1e-6
momentum=0.9

model=allcnn()
sgd=SGD(learning_rate=learning_rate,decay=weights_decay,momentum=momentum)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

epoch=350
batch_size=35

model.fit(x_train,Y_train,validation_data=(x_test,Y_test),epochs=epoch,batch_size=batch_size,verbose=1)

model.evaluate(x_test,Y_test,verbose=2)

classes=range(0,10)

names = ['airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']

class_labels=dict(zip(classes,names))
print(class_labels)

batch=x_test[100:109]
labels=np.argmax(Y_test[100:109],axis=1)
predections=model.predict(batch,verbose=2)
classifications=np.argmax(predections,axis=1)
print(classifications)


fig, axs = plt.subplots(3, 3, figsize = (15, 6))
fig.subplots_adjust(hspace = 1)
axs = axs.flatten()

for i, img in enumerate(batch):

    for key, value in class_labels.items():
        if classifications[i] == key:
            title = 'Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)
            
    axs[i].imshow(img)
    
plt.show()

