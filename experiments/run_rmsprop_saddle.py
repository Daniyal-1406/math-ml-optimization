import numpy as np
import matplotlib.pyplot as plt

from functions.saddle import loss, grad
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import Momentum
from optimizers.rmsprop import RMSProp


def run(steps=80):
    x0 = np.array([0.1, 0.1])

    gd = GradientDescent(lr=0.05)
    mom = Momentum(lr=0.05, beta=0.9)
    rms = RMSProp(epsilon=0.05, rho=0.9)

    x_gd = x0.copy()
    x_mom = x0.copy()
    x_rms = x0.copy()

    traj_gd = []
    traj_mom = []
    traj_rms = []

    for _ in range(steps):
        traj_gd.append(x_gd.copy())
        traj_mom.append(x_mom.copy())
        traj_rms.append(x_rms.copy())

        x_gd = gd.step(x_gd, grad(x_gd))
        x_mom = mom.step(x_mom, grad(x_mom))
        x_rms = rms.step(x_rms, grad(x_rms))

    traj_gd = np.array(traj_gd)
    traj_mom = np.array(traj_mom)
    traj_rms = np.array(traj_rms)

    plt.figure()
    plt.plot(traj_gd[:, 0], traj_gd[:, 1], label="GD", marker='o')
    plt.plot(traj_mom[:, 0], traj_mom[:, 1], label="Momentum", marker='x')
    plt.plot(traj_rms[:, 0], traj_rms[:, 1], label="RMSProp", marker='s')
    plt.title("Saddle Escape: GD vs Momentum vs RMSProp")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run()