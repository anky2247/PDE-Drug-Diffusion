
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Physical parameters
L = 0.01
N = 101
D = 1e-10
k = 1 / 1800
total_time = 1800.0

dx = L / (N - 1)
dy = dx

# Calculate stable time step
dt = min(5.0, 0.2 / (2 * D * (1/dx**2 + 1/dy**2)))
steps = int(np.ceil(total_time / dt))
dt = total_time / steps

# Create tissue grid
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Initial drug concentration: Gaussian distribution
center = L / 2
sigma = 0.0005

C = np.exp(
    -((X - center)**2 + (Y - center)**2)
    / (2 * sigma**2)
)

initial = C.copy()

# Solve the diffusion-reaction PDE
def solve_diffusion(concentration):

    snapshots = {0: concentration.copy()}

    target_times = [300, 900, 1800]
    target_steps = {
        round(t / dt): t for t in target_times
    }

    for step in range(1, steps + 1):

        padded = np.pad(concentration, 1, mode="edge")

        d2x = (
            padded[1:-1, 2:]
            - 2 * concentration
            + padded[1:-1, :-2]
        ) / dx**2

        d2y = (
            padded[2:, 1:-1]
            - 2 * concentration
            + padded[:-2, 1:-1]
        ) / dy**2

        concentration += dt * (
            D * (d2x + d2y) - k * concentration
        )

        if step in target_steps:
            snapshots[target_steps[step]] = concentration.copy()

    return snapshots


# Run simulation
# Run simulation
start = time.perf_counter()

snapshots = solve_diffusion(C.copy())

elapsed = time.perf_counter() - start

print("Simulation completed!")
print(f"Execution time: {elapsed:.4f} seconds")

# Display four heatmaps
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

times = [0, 300, 900, 1800]

for ax, t in zip(axes.flat, times):

    data = snapshots[t]

    plot = ax.imshow(
        data,
        extent=[0, L * 1000, 0, L * 1000],
        origin="lower",
        cmap="inferno"
    )

    ax.set_title(f"Time: {t // 60} minutes")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")

    fig.colorbar(plot, ax=ax)

fig.suptitle("Drug Diffusion Over Time")

plt.tight_layout()

Path("outputs").mkdir(exist_ok=True)

plt.savefig("outputs/diffusion_over_time.png", dpi=200)

plt.show()