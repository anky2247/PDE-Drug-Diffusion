import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pinn_model import PINN


# --------------------------------
# REPRODUCIBILITY
# --------------------------------

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cpu")

torch.set_num_threads(4)


# --------------------------------
# PHYSICAL PARAMETERS
# --------------------------------

L = 0.01
D = 1e-10
k = 1 / 1800
sigma = 0.0005
total_time = 1800.0

sigma_normalized = sigma / L

alpha = D * total_time / L**2
beta = k * total_time

print("Normalized diffusion coefficient:", alpha)
print("Normalized degradation coefficient:", beta)


# --------------------------------
# TRAINING SETTINGS
# --------------------------------

EPOCHS = 10000

N_COLLOCATION = 2048
N_INITIAL = 1024
N_BOUNDARY = 512

LEARNING_RATE = 0.001


# --------------------------------
# MODEL
# --------------------------------

model = PINN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

loss_history = []


# --------------------------------
# INITIAL CONDITION
# --------------------------------

def initial_condition(x, y):

    return torch.exp(
        -(
            (x - 0.5)**2
            + (y - 0.5)**2
        ) / (2 * sigma_normalized**2)
    )


# --------------------------------
# TRAINING LOOP
# --------------------------------

start_time = time.perf_counter()

print("\n===== PINN TRAINING STARTED =====")


for epoch in range(1, EPOCHS + 1):

    optimizer.zero_grad()


    # -----------------------------
    # 1. PDE COLLOCATION POINTS
    # -----------------------------

    n_focus = N_COLLOCATION // 2
    n_uniform = N_COLLOCATION - n_focus

    # Uniform points
    uniform_points = torch.rand(
        n_uniform, 3, device=device
    )

    # More points near the initial concentration peak
    focus_x = (
        0.5 + 0.15 * torch.randn(
            n_focus, 1, device=device
        )
    ).clamp(0, 1)

    focus_y = (
        0.5 + 0.15 * torch.randn(
            n_focus, 1, device=device
        )
    ).clamp(0, 1)

    focus_t = torch.rand(
        n_focus, 1, device=device
    )

    focus_points = torch.cat(
        [focus_x, focus_y, focus_t],
        dim=1
    )

    X_f = torch.cat(
        [uniform_points, focus_points],
        dim=0
    ).requires_grad_(True)


    # Network prediction
    u = model(X_f)


    # First derivatives
    gradients = torch.autograd.grad(
        u,
        X_f,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = gradients[:, 0:1]
    u_y = gradients[:, 1:2]
    u_t = gradients[:, 2:3]


    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x,
        X_f,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y,
        X_f,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 1:2]


    # PDE residual
    residual = (
        u_t
        - alpha * (u_xx + u_yy)
        + beta * u
    )

    loss_pde = torch.mean(residual**2)


    # -----------------------------
    # 2. INITIAL CONDITION
    # -----------------------------

    n_focus_ic = int(N_INITIAL * 0.7)
    n_uniform_ic = N_INITIAL - n_focus_ic

    xi_focus = (
        0.5 + 0.12 * torch.randn(
            n_focus_ic, 1, device=device
        )
    ).clamp(0, 1)

    yi_focus = (
        0.5 + 0.12 * torch.randn(
            n_focus_ic, 1, device=device
        )
    ).clamp(0, 1)

    xi_uniform = torch.rand(
        n_uniform_ic, 1, device=device
    )

    yi_uniform = torch.rand(
        n_uniform_ic, 1, device=device
    )

    xi = torch.cat([xi_focus, xi_uniform], dim=0)
    yi = torch.cat([yi_focus, yi_uniform], dim=0)

    ti = torch.zeros_like(xi)

    X_i = torch.cat([xi, yi, ti], dim=1)

    target_i = initial_condition(xi, yi)

    prediction_i = model(X_i)

    loss_initial = torch.mean(
        (prediction_i - target_i)**2
    )


    # -----------------------------
    # 3. NO-FLUX BOUNDARY CONDITION
    # -----------------------------

    t_boundary = torch.rand(
        N_BOUNDARY, 1, device=device
    )

    random_position = torch.rand(
        N_BOUNDARY, 1, device=device
    )

    # Left boundary: du/dx = 0
    left = torch.cat([
        torch.zeros_like(random_position),
        random_position,
        t_boundary
    ], dim=1).requires_grad_(True)

    # Right boundary: du/dx = 0
    right = torch.cat([
        torch.ones_like(random_position),
        random_position,
        t_boundary
    ], dim=1).requires_grad_(True)

    # Bottom boundary: du/dy = 0
    bottom = torch.cat([
        random_position,
        torch.zeros_like(random_position),
        t_boundary
    ], dim=1).requires_grad_(True)

    # Top boundary: du/dy = 0
    top = torch.cat([
        random_position,
        torch.ones_like(random_position),
        t_boundary
    ], dim=1).requires_grad_(True)


    def boundary_derivative(points, direction):

        output = model(points)

        gradient = torch.autograd.grad(
            output,
            points,
            grad_outputs=torch.ones_like(output),
            create_graph=True
        )[0]

        return gradient[:, direction:direction + 1]


    left_derivative = boundary_derivative(left, 0)
    right_derivative = boundary_derivative(right, 0)

    bottom_derivative = boundary_derivative(bottom, 1)
    top_derivative = boundary_derivative(top, 1)


    loss_boundary = (
        torch.mean(left_derivative**2)
        + torch.mean(right_derivative**2)
        + torch.mean(bottom_derivative**2)
        + torch.mean(top_derivative**2)
    )


    # -----------------------------
    # 4. TOTAL LOSS
    # -----------------------------

    loss = (
        5.0 * loss_pde
        + 100.0 * loss_initial
        + 5.0 * loss_boundary
    )

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())


    # Display progress
    if epoch % 100 == 0:

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Total Loss: {loss.item():.8e} | "
            f"PDE: {loss_pde.item():.8e} | "
            f"Initial: {loss_initial.item():.8e} | "
            f"Boundary: {loss_boundary.item():.8e}"
        )


elapsed = time.perf_counter() - start_time


# --------------------------------
# SAVE TRAINED MODEL
# --------------------------------

os.makedirs("outputs", exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "L": L,
    "D": D,
    "k": k,
    "sigma": sigma,
    "total_time": total_time
}

torch.save(
    checkpoint,
    "outputs/pinn_model.pt"
)


# --------------------------------
# SAVE LOSS GRAPH
# --------------------------------

plt.figure(figsize=(8, 5))

plt.semilogy(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("PINN Training Loss")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "outputs/pinn_training_loss.png",
    dpi=200
)

print("\n===== TRAINING COMPLETED =====")
print(f"Training time: {elapsed:.2f} seconds")
print("Model saved: outputs/pinn_model.pt")
print("Loss graph saved: outputs/pinn_training_loss.png")