
import numpy as np
import matplotlib.pyplot as plt
import time

# Physical parameters
L = 0.01
D = 1e-10
k = 1 / 1800
total_time = 1800.0
sigma = 0.0005

# Test different grid resolutions
grid_sizes = [51, 101, 201]

errors = []

def run_simulation(N):

    dx = L / (N - 1)
    dy = dx

    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)

    center = L / 2

    # Initial Gaussian concentration
    C = np.exp(
        -((X - center)**2 + (Y - center)**2)
        / (2 * sigma**2)
    )

    # Stable time step
    dt_limit = 0.2 / (
        2 * D * (1/dx**2 + 1/dy**2)
    )

    dt = min(5.0, dt_limit)

    steps = int(np.ceil(total_time / dt))
    dt = total_time / steps

    # Numerical solution
    for step in range(steps):

        padded = np.pad(C, 1, mode="edge")

        # Second derivative in X
        d2x = (
            padded[1:-1, 2:]
            - 2 * C
            + padded[1:-1, :-2]
        ) / dx**2

        # Second derivative in Y
        d2y = (
            padded[2:, 1:-1]
            - 2 * C
            + padded[:-2, 1:-1]
        ) / dy**2

        # Update concentration
        C += dt * (D * (d2x + d2y) - k * C)

    # Analytical solution for comparison
    variance = sigma**2 + 2 * D * total_time

    exact = (
        np.exp(-k * total_time)
        * sigma**2 / variance
        * np.exp(
            -((X - center)**2 + (Y - center)**2)
            / (2 * variance)
        )
    )

    # Relative L2 error
    error = np.linalg.norm(C - exact) / np.linalg.norm(exact)

    return error, steps


# Run grid accuracy experiment
print("\n===== GRID ACCURACY TEST =====")

start = time.perf_counter()

for N in grid_sizes:

    start = time.perf_counter()

    error, steps = run_simulation(N)

    grid_time = time.perf_counter() - start

    errors.append(error)

    print(f"Grid: {N} x {N}")
    print(f"Time steps: {steps}")
    print(f"Relative error: {error:.8f}")
    print(f"Execution time: {grid_time:.6f} seconds")
    print("-----------------------------")

elapsed = time.perf_counter() - start

print(f"Total execution time: {elapsed:.4f} seconds")

# Plot accuracy results
plt.figure(figsize=(8, 5))

plt.plot(grid_sizes, errors, marker="o")

plt.xlabel("Number of grid points per side")
plt.ylabel("Relative L2 error")
plt.title("Finite Difference Accuracy vs Grid Resolution")

plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/grid_accuracy.png", dpi=200)

