
import torch
import torch.nn as nn

class RhoNet2Q(nn.Module):
    def __init__(self, in_dim: int = 36, hidden=(256,256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        # diag(4) + 6 complex off-diagonals => 4 + 12 = 16 real params, but we output via two streams
        out_dim = 4 + 6  # we'll produce real diag + real off in one head, imag off in another head
        self.backbone = nn.Sequential(*layers)
        self.head_real = nn.Linear(last, out_dim)
        self.head_imag = nn.Linear(last, 6)
        self.softplus = nn.Softplus()

    def forward(self, x):
        b = self.backbone(x)
        r = self.head_real(b)
        im = self.head_imag(b)
        # diag
        d = self.softplus(r[..., :4])
        r_off = r[..., 4:]  # 6
        i_off = im          # 6
        B = x.shape[0]
        device = x.device
        L_real = torch.zeros((B,4,4), dtype=torch.float32, device=device)
        L_imag = torch.zeros((B,4,4), dtype=torch.float32, device=device)
        for i in range(4):
            L_real[:, i, i] = d[..., i]
        positions = [(1,0),(2,0),(2,1),(3,0),(3,1),(3,2)]
        for idx,(i,j) in enumerate(positions):
            L_real[:, i, j] = r_off[..., idx]
            L_imag[:, i, j] = i_off[..., idx]
        L = torch.complex(L_real, L_imag)
        A = L @ L.transpose(-1, -2).conj()
        tr = torch.real(torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)).clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1)
        rho = A / tr
        return rho
