import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data
hours = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
marks = np.array([35, 40, 50, 60, 70])

# Create model
model = LinearRegression()

# Train model
model.fit(hours, marks)

# Predict marks for 6 hours
prediction = model.predict([[6]])
print("Predicted marks for 6 hours of study:", prediction[0])

# Plot
plt.scatter(hours, marks, color='blue')
plt.plot(hours, model.predict(hours), color='red')
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.show()
