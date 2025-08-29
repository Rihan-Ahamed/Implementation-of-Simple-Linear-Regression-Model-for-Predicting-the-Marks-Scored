# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Rihan Ahamed.S
RegisterNumber:  212224040276
*/
```
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Gokul Nath R
RegisterNumber:  212224230077

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df= pd.read_csv('student_scores.csv')

df.head()
df.tail()

X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
y_pred

y_test

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
Head Values

<img width="359" height="268" alt="image" src="https://github.com/user-attachments/assets/25e9d46c-48a3-4084-8f4f-f5fdd4308954" />

Tail Values

<img width="338" height="267" alt="image" src="https://github.com/user-attachments/assets/fcb72c67-9d59-452e-9a6e-238f026644e3" />

X Values

<img width="501" height="637" alt="image" src="https://github.com/user-attachments/assets/62500d06-438a-431b-93e6-6ae58ac41572" />

Y Values

<img width="807" height="97" alt="image" src="https://github.com/user-attachments/assets/a0e2a3b7-7041-496b-bd38-13e495df6e3a" />

Predicted Values

<img width="805" height="79" alt="image" src="https://github.com/user-attachments/assets/da8a521f-abc7-46f0-8332-ff760acb2743" />

Actual Values

<img width="917" height="42" alt="image" src="https://github.com/user-attachments/assets/dc121b98-61a0-4212-afef-32c68f33a8db" />

Training set

<img width="935" height="610" alt="image" src="https://github.com/user-attachments/assets/69f04c20-9dfc-41a2-9238-473747f50f86" />

Testing set

<img width="920" height="599" alt="image" src="https://github.com/user-attachments/assets/3e20480b-e871-4e4a-9541-edea4e730ae7" />

 MSE, MAE and RMSE

<img width="534" height="93" alt="image" src="https://github.com/user-attachments/assets/1c0695a7-4038-4b25-9bf2-25358fa2e4fa" />











## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
