import numpy as np
import time
from statistics import median

# -------------------------------
# 1. SIMULATION PARAMETERS
# -------------------------------

L = 0.01
N = 201
D = 1e-10
k = 1 / 1800
total_time = 1800.0

dx = L / (N - 1)
dy = dx

# Stable time step
dt_limit = 0.2 / (
    2 * D * (1 / dx**2 + 1 / dy**2)
)

dt = min(5.0, dt_limit)

steps = int(np.ceil(total_time / dt))
dt = total_time / steps

# -------------------------------
# 2. INITIAL CONCENTRATION
# -------------------------------

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)

X, Y = np.meshgrid(x, y)

center = L / 2
sigma = 0.0005

initial = np.exp(
    -((X - center)**2 + (Y - center)**2)
    / (2 * sigma**2)
)


# -------------------------------
# 3. BASELINE SOLVER
# Creates new padded array every step
# -------------------------------

def baseline_solver(initial):

    C = initial.copy()

    for _ in range(steps):

        padded = np.pad(C, 1, mode="edge")

        d2x = (
            padded[1:-1, 2:]
            - 2 * C
            + padded[1:-1, :-2]
        ) / dx**2

        d2y = (
            padded[2:, 1:-1]
            - 2 * C
            + padded[:-2, 1:-1]
        ) / dy**2

        C += dt * (D * (d2x + d2y) - k * C)

    return C


# -------------------------------
# 4. OPTIMIZED SOLVER
# Reuses the same padded array
# -------------------------------

def optimized_solver(initial):

    C = initial.copy()

    # Allocate padded memory only once
    padded = np.empty((N + 2, N + 2))

    for _ in range(steps):

        # Copy concentration into the middle
        padded[1:-1, 1:-1] = C

        # Copy boundary values (no-flux boundary)
        padded[0, 1:-1] = C[0, :]
        padded[-1, 1:-1] = C[-1, :]

        padded[1:-1, 0] = C[:, 0]
        padded[1:-1, -1] = C[:, -1]

        d2x = (
            padded[1:-1, 2:]
            - 2 * C
            + padded[1:-1, :-2]
        ) / dx**2

        d2y = (
            padded[2:, 1:-1]
            - 2 * C
            + padded[:-2, 1:-1]
        ) / dy**2

        C += dt * (D * (d2x + d2y) - k * C)

    return C


# -------------------------------
# 5. WARM-UP
# -------------------------------

baseline_solver(initial)
optimized_solver(initial)


# -------------------------------
# 6. PERFORMANCE TEST
# -------------------------------

baseline_times = []
optimized_times = []

repeats = 5

for _ in range(repeats):

    start = time.perf_counter()
    baseline_result = baseline_solver(initial)
    baseline_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    optimized_result = optimized_solver(initial)
    optimized_times.append(time.perf_counter() - start)


# Use median to reduce timing noise
baseline_time = median(baseline_times)
optimized_time = median(optimized_times)


# -------------------------------
# 7. CHECK NUMERICAL ACCURACY
# -------------------------------

difference = (
    np.linalg.norm(baseline_result - optimized_result)
    / np.linalg.norm(baseline_result)
)


# -------------------------------
# 8. DISPLAY RESULTS
# -------------------------------

print("\n===== OPTIMIZATION RESULTS =====")

print(f"Grid size: {N} x {N}")
print(f"Time steps: {steps}")
print(f"Benchmark repetitions: {repeats}")

print("\nBaseline solver:")
print(f"Median time: {baseline_time:.6f} seconds")

print("\nOptimized solver:")
print(f"Median time: {optimized_time:.6f} seconds")

print("\nPerformance:")
print(f"Speedup: {baseline_time / optimized_time:.2f}x")

print("\nNumerical comparison:")
print(f"Relative difference: {difference:.12e}")

print("================================")