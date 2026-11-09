import numpy as np


def simulate(
    N=101,
    D=1e-10,
    k=1 / 1800,
    sigma=0.0005,
    total_time=1800.0,
    num_frames=41
):
    L = 0.01

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
        2 * D * (1 / dx**2 + 1 / dy**2)
    )

    dt = min(5.0, dt_limit)

    steps = int(np.ceil(total_time / dt))
    dt = total_time / steps

    # Steps at which we save heatmaps
    snapshot_steps = set(
        np.linspace(0, steps, num_frames, dtype=int).tolist()
    )

    frames = []
    times = []

    frames.append(C.copy())
    times.append(0.0)

    # Finite-difference time loop
    for step in range(1, steps + 1):

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

        if step in snapshot_steps:
            frames.append(C.copy())
            times.append(step * dt)

    frames = np.array(frames)
    times = np.array(times)

    # Concentration at the centre
    center_values = frames[:, N // 2, N // 2]

    # Integrated concentration over the domain
    total_mass = np.sum(frames, axis=(1, 2)) * dx * dy

    return {
        "x": x,
        "y": y,
        "times": times,
        "frames": frames,
        "center_values": center_values,
        "total_mass": total_mass,
        "dt": dt,
        "steps": steps,
        "N": N,
        "L": L
    }