import keras
from keras.models import Sequential
from keras.layers import *



#creating the model

model = Sequential()
model.add(Conv2D(20,(5,5),strides=2,activation = 'relu'))
model.add(MaxPooling2D(pool_size =(5,5)))
model.add(Conv2D(20,kernel_size=5,strides = 2,activation = 'relu'))
model.add(Flatten())
model.add(Dense(10000,activation = 'relu'))
model.add(Dense(1000,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='Softmax'))

model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics=['accuracy'])



#fitting the data
model.fit(TrainX,TrainY,epochs = 50, verbose = 1, )

