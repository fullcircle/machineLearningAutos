import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load dataset into a pandas DataFrame

# Create list of car models
models = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Audi', 'Mercedes', 'Hyundai', 'Kia']

# Create random data for hp and speed
hp = np.random.randint(low=100, high=500, size=1000)
speed = np.random.randint(low=100, high=300, size=1000)

# Create random model data
model = [models[i % len(models)] for i in range(1000)]

# Create pandas dataframe
df = pd.DataFrame({'model': model, 'hp': hp, 'speed': speed})

# df = pd.read_csv('cars.csv')
df = pd.get_dummies(df, columns=['model'])
print(df)
# Split dataset into training and testing sets
# Split dataset into training and testing sets
X = df.drop('speed', axis=1)
y = df['speed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict speed for test dataset
y_pred = model.predict(X_test)

# Evaluate the model
score = model.score(X_test, y_test)
print("R-squared score:", score)