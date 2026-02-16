import numpy as np
import matplotlib.pyplot as plt

from functions.rosenbrock import loss, grad
from optimizers.gradient_descent import GradientDescent
from optimizers.momentum import Momentum
from optimizers.rmsprop import RMSProp


def run(steps=5000):
    x0 = np.array([-1.5, 2.0])

    gd = GradientDescent(lr=0.001)
    mom = Momentum(lr=0.001, beta=0.9)
    rms = RMSProp(epsilon=0.001, rho=0.9)

    x_gd = x0.copy()
    x_mom = x0.copy()
    x_rms = x0.copy()

    loss_gd = []
    loss_mom = []
    loss_rms = []

    for _ in range(steps):
        loss_gd.append(loss(x_gd))
        loss_mom.append(loss(x_mom))
        loss_rms.append(loss(x_rms))

        x_gd = gd.step(x_gd, grad(x_gd))
        x_mom = mom.step(x_mom, grad(x_mom))
        x_rms = rms.step(x_rms, grad(x_rms))

    plt.figure()
    plt.plot(loss_gd, label="GD")
    plt.plot(loss_mom, label="Momentum")
    plt.plot(loss_rms, label="RMSProp")
    plt.yscale("log")
    plt.title("Rosenbrock: GD vs Momentum vs RMSProp")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run()