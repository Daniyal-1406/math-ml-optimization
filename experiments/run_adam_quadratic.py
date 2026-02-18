import numpy as np
import matplotlib.pyplot as plt

from functions.quadratic import make_quadratic
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import Momentum
from optimizers.rmsprop import RMSProp
from optimizers.Adam import Adam


def run(condition_number=100, steps=100):
    loss, grad, A = make_quadratic(condition_number)

    eigenvalues = np.linalg.eigvals(A)
    L = max(eigenvalues)

    x0 = np.array([5.0, 5.0])

    gd = GradientDescent(lr=1 / L)
    mom = Momentum(lr=1 / L, beta=0.9)
    rms = RMSProp(epsilon=0.05, rho=0.9)
    adam = Adam(lr=0.05)

    x_gd = x0.copy()
    x_mom = x0.copy()
    x_rms = x0.copy()
    x_adam = x0.copy()

    loss_gd, loss_mom, loss_rms, loss_adam = [], [], [], []

    for _ in range(steps):
        loss_gd.append(loss(x_gd))
        loss_mom.append(loss(x_mom))
        loss_rms.append(loss(x_rms))
        loss_adam.append(loss(x_adam))

        x_gd = gd.step(x_gd, grad(x_gd))
        x_mom = mom.step(x_mom, grad(x_mom))
        x_rms = rms.step(x_rms, grad(x_rms))
        x_adam = adam.step(x_adam, grad(x_adam))

    plt.figure()
    plt.plot(loss_gd, label="GD")
    plt.plot(loss_mom, label="Momentum")
    plt.plot(loss_rms, label="RMSProp")
    plt.plot(loss_adam, label="Adam")
    plt.yscale("log")
    plt.title("Quadratic: GD vs Momentum vs RMSProp vs Adam")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run()