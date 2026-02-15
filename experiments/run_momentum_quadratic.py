import numpy as np
import matplotlib.pyplot as plt

from functions.quadratic import make_quadratic
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import Momentum


def run(condition_number=100, steps=80):
    loss, grad, A = make_quadratic(condition_number)

    eigenvalues = np.linalg.eigvals(A)
    L = max(eigenvalues)

    x0 = np.array([5.0, 5.0])

    gd = GradientDescent(lr=1 / L)
    mom = Momentum(lr=1 / L, beta=0.9)

    x_gd = x0.copy()
    x_mom = x0.copy()

    loss_gd = []
    loss_mom = []

    for _ in range(steps):
        loss_gd.append(loss(x_gd))
        loss_mom.append(loss(x_mom))

        g_gd = grad(x_gd)
        g_mom = grad(x_mom)

        x_gd = gd.step(x_gd, g_gd)
        x_mom = mom.step(x_mom, g_mom)

    plt.figure()
    plt.plot(loss_gd, label="Gradient Descent")
    plt.plot(loss_mom, label="Momentum")
    plt.yscale("log")
    plt.title(f"GD vs Momentum (κ={condition_number})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run(condition_number=100)