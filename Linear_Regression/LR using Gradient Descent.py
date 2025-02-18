import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
data = pd.read_csv('Advertising.csv')

# Check and remove missing values
data = data.dropna()

# Normalize data to prevent large gradient values
data['TV'] = data['TV'] / data['TV'].max()
data['Sales'] = data['Sales'] / data['Sales'].max()

# Loss function
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i]['TV']
        y = points.iloc[i]['Sales']
        total_error += (y - (m * x + b)) ** 2
    return total_error / len(points)

# Gradient descent function
def gradient_descent(m_cur, b_cur, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i]['TV']
        y = points.iloc[i]['Sales']
        m_gradient += -(2/n) * x * (y - (m_cur * x + b_cur))
        b_gradient += -(2/n) * (y - (m_cur * x + b_cur))

    m = m_cur - m_gradient * L
    b = b_cur - b_gradient * L

    return m, b

# Initialize parameters
m, b = 0, 0
L = 0.01  # Try reducing to 0.0001 if still NaN
epochs = 1000

# Gradient descent loop
for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
    if np.isnan(m) or np.isnan(b):  # Stop if NaN occurs
        print("Gradient descent diverged! Try reducing L.")
        break

# Print final values
print(f'm = {m}')
print(f'b = {b}')

# Plot results
plt.scatter(data.TV, data.Sales, color='black')
x_vals = np.linspace(data.TV.min(), data.TV.max(), 100)
y_vals = m * x_vals + b
plt.plot(x_vals, y_vals, color='red')

# Labels
plt.xlabel('TV Advertising Budget (Normalized)')
plt.ylabel('Sales (Normalized)')
plt.title('Linear Regression using Gradient Descent')

plt.show()
