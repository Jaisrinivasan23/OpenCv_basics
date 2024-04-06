from numpy import loadtxt
from keras.models import load_model

dataset = loadtxt('Datasets\diabetes.csv',delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

model = load_model('diabetes_model.h5')

predictions = model.predict(x)
for i in range(10,15):
    print('%s => %d (expected %d)' % (x[i].tolist(),predictions[i],y[i]))
