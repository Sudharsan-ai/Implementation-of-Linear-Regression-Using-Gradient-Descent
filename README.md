# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load and preprocess data**: Read CSV data, extract features (`X`) and target (`y`), and convert them to float.
2. **Normalize features**: Apply `StandardScaler` to scale both features and target for better training performance.
3. **Define model**: Implement gradient descent-based linear regression with bias term added to input features.
4. **Train model**: Iterate to minimize error and update model parameters (`theta`) using gradient descent.
5. **Predict new value**: Scale the new input data, apply the trained model, and get the scaled prediction.
6. **Inverse scale result**: Convert the scaled prediction back to the original scale and print it.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SUDHARSAN S
RegisterNumber:  212224040334
*\
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

# Load dataset
data = pd.read_csv("50_Startups.csv")

# Use only numeric columns (drop 'State')
X = data.iloc[:, [0, 1, 2]].values  # R&D, Administration, Marketing Spend
y = data.iloc[:, -1].values        # Profit

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Linear Regression using Gradient Descent
model = SGDRegressor(max_iter=10000, learning_rate='constant', eta0=0.01)
model.fit(X_scaled, y)

# New data for prediction
new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_scaled = scaler.transform(new_data)

# Predict profit
prediction = model.predict(new_scaled)
print("Predicted Profit:", prediction[0])

```

## Output:
![image](https://github.com/user-attachments/assets/053d5d73-e837-47d4-a8a2-39bd7c2c429e)

![image](https://github.com/user-attachments/assets/221d7eb2-9871-4d92-b1a1-a8e30b4837d5)

![image](https://github.com/user-attachments/assets/97895db7-8c77-4348-bc56-4240fb1165cb)

![image](https://github.com/user-attachments/assets/3f38a072-a4a8-4925-ade2-e0c7b700c0b3)

![image](https://github.com/user-attachments/assets/c0f92d6b-9b00-47a0-a9a5-12920b1ba734)

![image](https://github.com/user-attachments/assets/210d0076-58b5-4f67-9b56-c485b924a9a9)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
