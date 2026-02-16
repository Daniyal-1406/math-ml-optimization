import numpy as np
import matplotlib.pyplot as plt

from functions.quadratic import make_quadratic
from optimizers.rmsprop import RMSProp


def run(condition_number=100, steps=80):
    loss, grad, A = make_quadratic(condition_number)

    eigenvalues = np.linalg.eigvals(A)
    L = max(eigenvalues)

    x0 = np.array([5.0, 5.0])

    rhos = [0.5, 0.9, 0.99]

    plt.figure()

    for rho in rhos:
        optimizer = RMSProp(epsilon=1/L, rho=rho)

        x = x0.copy()
        losses = []

        for _ in range(steps):
            losses.append(loss(x))
            g = grad(x)
            x = optimizer.step(x, g)

        plt.plot(losses, label=f"rho={rho}")

    plt.yscale("log")
    plt.title(f"RMSProp: Effect of rho (κ={condition_number})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run()