import numpy as np
import matplotlib.pyplot as plt
from optimizers.gradient_descent import GradientDescent
from functions.quadratic import make_quadratic


def run_experiment(condition_number=10, learning_rates=None, steps=60):
    if learning_rates is None:
        learning_rates = [0.01, 0.05, 0.1, 0.2]

    loss, grad, A = make_quadratic(condition_number)

    eigenvalues = np.linalg.eigvals(A)
    L = max(eigenvalues)
    mu = min(eigenvalues)
    kappa = L / mu

    print(f"\nCondition Number κ = {kappa}")
    print(f"L = {L}, μ = {mu}")

    plt.figure()

    for lr in learning_rates:
        x = np.array([5.0, 5.0])
        optimizer = GradientDescent(lr=lr)

        loss_history = []

        for _ in range(steps):
            loss_history.append(loss(x))
            g = grad(x)
            x = optimizer.step(x, g)

        plt.plot(loss_history, label=f"lr={lr}")

    plt.title(f"Loss vs Iteration (κ={condition_number})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_experiment(
        condition_number=10,
        learning_rates=[0.01, 0.05, 0.1, 0.2, 0.3]
    )