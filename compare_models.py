import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pinn_model import PINN, predict_grid
from pde_solver import simulate


# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------

device = torch.device("cpu")

checkpoint = torch.load(
    "outputs/pinn_model.pt",
    map_location=device,
    weights_only=True
)

model = PINN().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# --------------------------------
# LOAD PARAMETERS
# --------------------------------

L = checkpoint["L"]
D = checkpoint["D"]
k = checkpoint["k"]
sigma = checkpoint["sigma"]
total_time = checkpoint["total_time"]


# --------------------------------
# RUN FINITE DIFFERENCE SOLVER
# --------------------------------

N = 101

fd_result = simulate(
    N=N,
    D=D,
    k=k,
    sigma=sigma,
    total_time=total_time,
    num_frames=5
)

x = fd_result["x"]
y = fd_result["y"]

times = fd_result["times"]
fd_frames = fd_result["frames"]


# --------------------------------
# PINN PREDICTIONS
# --------------------------------

pinn_frames = []

for t in times:

    prediction = predict_grid(
        model,
        x,
        y,
        t,
        L,
        total_time
    )

    pinn_frames.append(prediction)

pinn_frames = np.array(pinn_frames)


# --------------------------------
# ERROR CALCULATION
# --------------------------------

print("\n===== PINN VS FINITE DIFFERENCE =====")

errors = []

for i, t in enumerate(times):

    fd = fd_frames[i]
    pinn = pinn_frames[i]

    relative_error = (
        np.linalg.norm(fd - pinn)
        / np.linalg.norm(fd)
    )

    errors.append(relative_error)

    print(f"Time: {t:.2f} seconds")
    print(f"Relative L2 error: {relative_error:.8f}")
    print("-----------------------------------")


# --------------------------------
# FINAL-TIME COMPARISON
# --------------------------------

fd_final = fd_frames[-1]
pinn_final = pinn_frames[-1]

absolute_error = np.abs(
    fd_final - pinn_final
)

final_error = errors[-1]

print("\nFinal relative L2 error:", final_error)


# --------------------------------
# VISUAL COMPARISON
# --------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5)
)

extent = [
    x[0] * 1000,
    x[-1] * 1000,
    y[0] * 1000,
    y[-1] * 1000
]

im1 = axes[0].imshow(
    fd_final,
    origin="lower",
    extent=extent,
    cmap="inferno"
)

axes[0].set_title("Finite Difference")

axes[0].set_xlabel("X (mm)")
axes[0].set_ylabel("Y (mm)")

fig.colorbar(im1, ax=axes[0])


im2 = axes[1].imshow(
    pinn_final,
    origin="lower",
    extent=extent,
    cmap="inferno"
)

axes[1].set_title("PINN Prediction")

axes[1].set_xlabel("X (mm)")
axes[1].set_ylabel("Y (mm)")

fig.colorbar(im2, ax=axes[1])


im3 = axes[2].imshow(
    absolute_error,
    origin="lower",
    extent=extent,
    cmap="viridis"
)

axes[2].set_title("Absolute Error")

axes[2].set_xlabel("X (mm)")
axes[2].set_ylabel("Y (mm)")

fig.colorbar(im3, ax=axes[2])


plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/model_comparison.png",
    dpi=200
)

print("\nComparison image saved:")
print("outputs/model_comparison.png")