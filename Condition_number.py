import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Function (High Condition Number)
# f(x, y) = 0.5 * (x^2 + 10 * y^2)
# The Condition Number is kappa = 10 / 1 = 10.
def f(x, y):
    return 0.5 * (x**2 + 10 * y**2)

# Gradient: [x, 10y]
def grad_f(x, y):
    return np.array([x, 10 * y])

# 2. Simulate Gradient Descent
path = []
x_curr = np.array([3.5, 1.5]) # Start at a point that triggers zig-zag
path.append(x_curr.copy())

# Learning Rate (Step Size)
# Max stable rate is 2/L = 2/10 = 0.2.
# We pick 0.18, which is close to the limit, causing heavy oscillation.
learning_rate = 0.18
iterations = 20

for _ in range(iterations):
    grad = grad_f(x_curr[0], x_curr[1])
    x_curr = x_curr - learning_rate * grad
    path.append(x_curr.copy())

path = np.array(path)

# 3. Setup the Contour Plot
x_range = np.linspace(-4, 4, 100)
y_range = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# 4. Create the Visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Contours (The "Valley")
# Using logspace for levels to see the valley floor better
ax.contour(X, Y, Z, levels=np.logspace(-1, 2, 20), cmap='viridis', alpha=0.5)

# Plot the Path (The "Zig-Zag")
ax.plot(path[:, 0], path[:, 1], 'r-o', label='Gradient Descent Path', markersize=4, linewidth=1.5)

# Annotations
ax.annotate('Steep Gradient in Y', xy=(path[0][0], path[0][1]), xytext=(path[0][0]+0.5, path[0][1]+0.5),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.set_title(r'High Condition Number ($\kappa = 10$) $\rightarrow$ Zig-Zag Behavior')
ax.set_xlabel('x (Flat direction)')
ax.set_ylabel('y (Steep direction)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_aspect('equal') # Crucial to see the elongation

plt.tight_layout()
plt.show()