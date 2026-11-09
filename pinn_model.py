import torch
import torch.nn as nn


class PINN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(3, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 1)
        )

        # Xavier initialization
        for layer in self.network:

            if isinstance(layer, nn.Linear):

                nn.init.xavier_normal_(layer.weight)

                nn.init.zeros_(layer.bias)

    def forward(self, x):

        return self.network(x)


def predict_grid(model, x, y, time_seconds, L, total_time):

    import numpy as np

    X, Y = np.meshgrid(x / L, y / L)

    tau = time_seconds / total_time

    points = np.column_stack([
        X.ravel(),
        Y.ravel(),
        np.full(X.size, tau)
    ])

    device = next(model.parameters()).device

    predictions = []

    model.eval()

    with torch.no_grad():

        for start in range(0, len(points), 8192):

            batch = torch.tensor(
                points[start:start + 8192],
                dtype=torch.float32,
                device=device
            )

            output = model(batch)

            predictions.append(
                output.cpu().numpy()
            )

    predictions = np.concatenate(predictions)

    return predictions.reshape(X.shape)
    