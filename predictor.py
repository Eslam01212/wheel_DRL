import numpy as np
import torch
import torch.nn as nn

class WheelMLP(nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, x): return self.net(x)

class WheelRuntimePredictor:
    def __init__(self, ckpt_path="wheel_predictor.pt", device="cpu"):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.x_mean = np.asarray(ck["x_mean"], dtype=np.float32).reshape(-1)
        self.x_std  = np.asarray(ck["x_std"],  dtype=np.float32).reshape(-1)
        self.device = device
        self.m = WheelMLP(in_dim=len(self.x_mean)).to(device).eval()
        self.m.load_state_dict(ck["state_dict"])

    @torch.no_grad()
    def __call__(self, x8: np.ndarray) -> np.ndarray:
        x8 = np.asarray(x8, dtype=np.float32).reshape(-1)
        x8 = np.nan_to_num(x8, nan=0.0, posinf=0.0, neginf=0.0)

        x = (x8 - self.x_mean) / np.maximum(self.x_std, 1e-6)
        x = np.clip(x, -10.0, 10.0)

        X = torch.from_numpy(x).to(self.device).unsqueeze(0)
        rad = self.m(X)[0]
        rad = torch.nan_to_num(rad, nan=1.0, posinf=1.5, neginf=0.5)

        # keep same mapping as your old runtime: clamp [0.5,1.5] then -> [-1,1]
        rad = torch.clamp(rad, 0.2, 1.8)   # match dataset sampling range
        rad = (rad - 1.0) / 0.8           # half-range = (1.8-0.2)/2 = 0.8

        return rad.detach().cpu().numpy().astype(np.float32)
