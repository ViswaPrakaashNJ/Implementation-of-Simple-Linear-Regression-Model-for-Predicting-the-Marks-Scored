# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1. Start the program.

STEP 2.Import the standard Libraries.

STEP 3.Set variables for assigning dataset values.

STEP 4.Import linear regression from sklearn.

STEP 5.Assign the points for representing in the graph.

STEP 6.Predict the regression for marks by using the representation of the graph.

STEP 7.End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VISWA PRAKAASH N J
RegisterNumber:  212223040246
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()
df.tail()
```
```
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
df.head()

![305892262-b167b189-955e-4a58-98ed-81521b6a2307](https://github.com/user-attachments/assets/587e73b3-da66-4ab8-b80a-6d08a4585966)

df.tail()

![305892379-6e5d3991-a656-40dd-b590-f0cde57a9df7](https://github.com/user-attachments/assets/9681ced8-3c40-4ebd-bb87-1388245c25a2)

Array value of X

![305892578-cbdade85-e2f8-4033-99ca-c150a2540f4d](https://github.com/user-attachments/assets/51f2c97e-52e3-411f-a478-c6e89671fdf4)

Array value of Y

![305892769-bd500e67-4f89-4d8e-ac02-7a30adb63ea3](https://github.com/user-attachments/assets/c88cb1a1-978c-41fb-b09f-8e3f617276f6)

Values of y prediction

![305893048-0fb49543-18a3-462b-83e8-48da46913b68](https://github.com/user-attachments/assets/14565c51-41cf-4125-80dd-84174db5640a)

Array values of Y test

![305893180-b84929fa-8bad-4fdd-8c4c-9c4b27499f28](https://github.com/user-attachments/assets/ee4f5f85-7310-4cf1-8576-ed26a9263d9e)

Training set graph

![305894045-67edac83-7c77-495f-8a34-0f5d5a7659df](https://github.com/user-attachments/assets/c3e88499-6d02-477b-b36d-2098d05952b6)

Test set graph

![305894600-3fc3a7b4-9592-44cb-bd08-0db9b71100c7](https://github.com/user-attachments/assets/50114bf5-ed34-4ff2-a07e-8df2072df834)

Values of MSE,MAE and RMSE

![305894805-be305b14-73bf-4a15-ae7f-7ade05dfec07](https://github.com/user-attachments/assets/bb6b3976-dca3-4534-916a-0de89d0359fa)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
