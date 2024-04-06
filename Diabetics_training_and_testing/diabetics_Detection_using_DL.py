from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt('Datasets\diabetes.csv',delimiter=',')

X = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model Training
model.fit(X,y,epochs=50,batch_size=10)

Accuracy = model.evaluate(X,y)
print('Accuracy: %.2f' % (Accuracy[1]*100))

model.save('diabetes_model.h5')