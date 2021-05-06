
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM,Dropout
from keras.layers import Dense

class Build:
    def __init__(self,X , y):
        self.X = X
        self.y = y
        self.y_flat = np.argmax(y, axis =1 )
        self.input_shape = (X.shape[1], X.shape[2],1)
        self.class_weight = compute_class_weight('balanced',np.unique(self.y_flat), self.y_flat)


    def creat_conv_model(self):
        model = Sequential()
        model.add(Conv2D(16 ,(3,3), activation='relu', strides= (1,1), padding='same',input_shape=self.input_shape))
        model.add(Conv2D(32,(3,3),activation='relu', strides=(1,1), padding='same'))
        model.add(Conv2D(64,(3,3),activation='relu', strides=(1,1), padding='same'))
        model.add(Conv2D(128,(3,3),activation='relu', strides=(1,1), padding='same'))
        model.add(MaxPool2D((2,2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(6, activation='softmax'))
        model.summary()

        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
        return model


