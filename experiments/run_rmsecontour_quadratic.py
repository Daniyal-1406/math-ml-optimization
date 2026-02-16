import numpy as np
import matplotlib.pyplot as plt

from functions.quadratic import make_quadratic
from optimizers.rmsprop import RMSProp


def run(condition_number=100, steps=60):

    loss, grad, A = make_quadratic(condition_number)

    eigenvalues = np.linalg.eigvals(A)
    L = max(eigenvalues)

    x0 = np.array([5.0, 5.0])

    rms = RMSProp(epsilon=0.1, rho=0.9)

    x = x0.copy()
    trajectory = []

    for _ in range(steps):
        trajectory.append(x.copy())
        x = rms.step(x, grad(x))

    trajectory = np.array(trajectory)

    # ----- Create Contour -----
    x1 = np.linspace(-6, 6, 400)
    x2 = np.linspace(-6, 6, 400)
    X1, X2 = np.meshgrid(x1, x2)

    Z = 0.5 * (A[0, 0] * X1**2 + 2 * A[0, 1] * X1 * X2 + A[1, 1] * X2**2)

    plt.figure()
    plt.contour(X1, X2, Z, levels=30)

    # ----- Plot RMSProp trajectory -----
    plt.plot(trajectory[:, 0], trajectory[:, 1],
             color='red', marker='o')

    plt.title("RMSProp on Quadratic")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run(condition_number=100)