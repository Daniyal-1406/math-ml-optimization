import numpy as np
import matplotlib.pyplot as plt

from functions.saddle import loss, grad
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import Momentum


def run(steps=60):
    x0 = np.array([0.1, 0.1])

    gd = GradientDescent(lr=0.05)
    mom = Momentum(lr=0.05, beta=0.9)

    x_gd = x0.copy()
    x_mom = x0.copy()

    traj_gd = []
    traj_mom = []

    for _ in range(steps):
        traj_gd.append(x_gd.copy())
        traj_mom.append(x_mom.copy())

        x_gd = gd.step(x_gd, grad(x_gd))
        x_mom = mom.step(x_mom, grad(x_mom))

    traj_gd = np.array(traj_gd)
    traj_mom = np.array(traj_mom)

    plt.figure()
    plt.plot(traj_gd[:, 0], traj_gd[:, 1], label="GD", marker='o')
    plt.plot(traj_mom[:, 0], traj_mom[:, 1], label="Momentum", marker='x')
    plt.title("Saddle Escape: GD vs Momentum")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run()