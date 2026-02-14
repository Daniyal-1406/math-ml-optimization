import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# 1. Define the Function (Simple Convex Bowl)
def f(x, y):
    return 0.5 * (x**2 + y**2)  # f(x) = 1/2 ||x||^2

# Gradient of f at (x, y)
def grad_f(x, y):
    return np.array([x, y])

# 2. Setup the Grid
x = np.linspace(-2, 2, 30)
y = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x, y)
Z_func = f(X, Y)

# 3. Pick a Reference Point (x0)
x0, y0 = 1.0, 0.5
z0 = f(x0, y0)
g0 = grad_f(x0, y0)  # Gradient at x0

# 4. Calculate Surfaces
# Tangent Plane (Linear Lower Bound)
# L(y) = f(x) + grad(x)^T (y-x)
Z_tangent = z0 + g0[0]*(X - x0) + g0[1]*(Y - y0)

# Lipschitz Quadratic Upper Bound (The "Ceiling")
# Q_upper(y) = Tangent + (L/2) ||y-x||^2
# Actual curvature is 1. We choose L=2.5 to show a loose upper bound.
L = 2.5
dist_sq = (X - x0)**2 + (Y - y0)**2
Z_upper = Z_tangent + (L / 2) * dist_sq

# Strong Convexity Quadratic Lower Bound (The "Floor")
# Q_lower(y) = Tangent + (mu/2) ||y-x||^2
# Actual curvature is 1. We choose mu=0.5 to show a loose lower bound.
mu = 0.5
Z_lower = Z_tangent + (mu / 2) * dist_sq

# 5. Plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the Lipschitz Upper Bound (Red Wireframe - The "Ceiling")
ax.plot_wireframe(X, Y, Z_upper, color='red', alpha=0.4, rstride=3, cstride=3)

# Plot the actual function (Blue Surface)
# Making it slightly transparent to see the sandwich
ax.plot_surface(X, Y, Z_func, alpha=0.3, color='blue', edgecolor='none')

# Plot the Strong Convexity Lower Bound (Purple Wireframe - The "Floor")
# This sits ABOVE the tangent but BELOW the function
ax.plot_wireframe(X, Y, Z_lower, color='purple', alpha=0.6, rstride=3, cstride=3)

# Plot the Tangent Plane (Green Surface - Flat)
# This is the "weakest" lower bound
ax.plot_surface(X, Y, Z_tangent, alpha=0.2, color='green')

# Mark the point x0
ax.scatter(x0, y0, z0, color='black', s=100, label='Point $x_0$')

# Labels and Styling
ax.set_title('Sandwiching the Function:\nStrong Convexity vs. Lipschitz Gradient')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x)')

# Set view to see the separation clearly
ax.view_init(elev=20, azim=-45)
ax.set_zlim(-2, 5) # Limit Z axis to keep view focused

# Add a custom legend
legend_elements = [
    Line2D([0], [0], color='red', lw=2, label=f'Upper Bound (Lipschitz L={L})'),
    Line2D([0], [0], color='blue', lw=4, alpha=0.3, label='Actual Function f(x)'),
    Line2D([0], [0], color='purple', lw=2, label=f'Lower Bound (Strong Conv mu={mu})'),
    Line2D([0], [0], color='green', lw=4, alpha=0.2, label='Tangent Plane (Linear)')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()