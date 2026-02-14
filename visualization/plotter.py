import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(history, loss_fn, title="Optimization Trajectory"):
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]

    # Create contour grid
    x_vals = np.linspace(-6, 6, 200)
    y_vals = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 0.5 * (X**2 + 10 * Y**2)

    plt.figure()
    plt.contour(X, Y, Z, levels=20)
    plt.plot(xs, ys, marker='o', color='red')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.grid(True)
    plt.show()