import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df = pd.read_csv('hearts.csv')

df['Sex'] = le.fit_transform(df['Sex'])
df['ChestPain'] = le.fit_transform(df['ChestPain'])
df['RestECG'] = le.fit_transform(df['RestECG'])
df['ExAng'] = le.fit_transform(df['ExAng'])
df['Slope'] = le.fit_transform(df['Slope'])

x=df.drop('AHD',axis=1)
y=df['AHD']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

testPred = model.predict([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]])
if testPred == 1:
    print("Heart Disease")
else:
    print("No Heart Disease")

testPred2 = model.predict([[55,0,2,132,342,0,0,166,0,1.2,1,0,2]])
if testPred2 == 1:
    print("Heart Disease")
else:
    print("No Heart Disease")
