import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

def define_model():
    model = Sequential()
    model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def test_model(model):
    model.fit(x_train,y_train,batch_size=32,nb_epoch=10,verbose=1)
    score = model.evaluate(x_test,y_test,verbose=0)
    print(model.metrics[0],score[1]*100)


model = define_model()
test_model(model)

