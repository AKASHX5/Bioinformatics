import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing, cross_validation,neighbors
df=pd.read_csv('Test.data')
#df.index=df['UserId']
df.replace('?',-99999,inplace=True)
df.drop(['Id'],1,inplace=True)

x=np.array(df.drop(['class'],1))
y=np.array(df['class'])



x_train,y_train,x_test,y_test=cross_validation.train_test_split(x,y,test_size=0.2)

#df = df.fillna(method='ffill')

clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print (accuracy)
