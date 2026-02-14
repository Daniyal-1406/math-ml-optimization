import numpy as np
import matplotlib.pyplot as plt
from functions.saddle import loss, grad
from optimizers.gradient_descent import GradientDescent


def run():
    x = np.array([0.1, 0.1])  # start near saddle
    lr = 0.05
    steps = 60

    optimizer = GradientDescent(lr=lr)

    trajectory = []

    for _ in range(steps):
        trajectory.append(x.copy())
        g = grad(x)
        x = optimizer.step(x, g)

    trajectory = np.array(trajectory)

    plt.figure()
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
    plt.title("Gradient Descent on Saddle Surface")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run()