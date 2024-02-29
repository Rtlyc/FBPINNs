
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np

# Constants
m = 1.0  # mass
d = 2
w0 = 80
mu = 2 * m * d  # damping
k = w0 * w0 * m  # spring constant
u0 = 1.0  # initial displacement
v0 = 0.0  # initial velocity

def cosine_window(x, xmin, xmax, device):
    """Compute the cosine window function for smooth transitions."""
    # Scale and shift x to [-1, 1]
    x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1
    # Apply cosine window function
    window = (1 + torch.cos(x_scaled * torch.pi)) / 2
    # Zero out values outside [xmin, xmax]
    window = torch.where((x >= xmin) & (x <= xmax), window, torch.tensor(0.0, device=device))
    return window


class SpringSubNN(nn.Module):
    def __init__(self, device):
        super(SpringSubNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).to(device)

    def forward(self, t):
        return self.net(t)


class SpringNN(nn.Module):
    def __init__(self, device, subdomain_xs, subdomain_ws, unnorm):
        super(SpringNN, self).__init__()
        self.subnets = nn.ModuleList([SpringSubNN(device) for _ in range(len(subdomain_xs[0]))])
        self.subnet_ranges = [(x-w/2, x+w/2) for x, w in zip(subdomain_xs[0], subdomain_ws[0])]
        self.unnorm = unnorm
        self.device = device

    def forward(self, t):
        outputs = torch.zeros_like(t)
        for i, subnet in enumerate(self.subnets):
            xmin, xmax = self.subnet_ranges[i]
            weight = cosine_window(t, xmin, xmax, device)
            outputs += weight * subnet(t)
        return outputs

def loss_fn(model, t):
    u = model(t)
    ones = torch.ones_like(u)
    u_t = grad(u, t, grad_outputs=ones, create_graph=True)[0]
    u_tt = grad(u_t, t, grad_outputs=ones, create_graph=True)[0]
    
    de_loss = (m * u_tt + mu * u_t + k * u).pow(2).mean()
    
    t0 = torch.tensor([[0.0]], device=model.device, requires_grad=True)
    u0_ = model(t0)
    ut0_ = grad(u0_, t0, create_graph=True)[0]
    
    ic_loss_u = 1e6 * (u0_ - u0).pow(2)
    ic_loss_v = 1e2 * (ut0_ - v0).pow(2)
    
    return de_loss + ic_loss_u + ic_loss_v

def exact_solution(x_batch, device):
    w = torch.sqrt(torch.tensor([w0**2 - d**2], device=device))
    phi = torch.arctan(torch.tensor([-d/w], device=device))
    A = 1 / (2 * torch.cos(phi))
    cos = torch.cos(phi + w * x_batch)
    exp = torch.exp(-d * x_batch)
    u = exp * 2 * A * cos
    return u

def viz(model, device):
    t_vals = torch.linspace(0, 1, steps=200, device=device).reshape(-1, 1)
    with torch.no_grad():
        predictions = model(t_vals).cpu().numpy()

    sol = exact_solution(t_vals.squeeze(), device).cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals.cpu().numpy(), predictions, label='NN Prediction')
    plt.plot(t_vals.cpu().numpy(), sol, label='Exact Solution')
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming decomposition logic to define subdomains
    decomposition_init_kwargs = {
        'subdomain_xs': [np.linspace(0, 1, 15)],  # 15 equally spaced subdomains
        'subdomain_ws': [0.15 * np.ones((15,))],  # with widths of 0.15
        'unnorm': (0., 1.),  # define unnormalisation of the subdomain networks
    }

    model = SpringNN(device, **decomposition_init_kwargs)
    # subnet_ranges = [(0, 0.4), (0.3, 0.7), (0.6, 1.0)]
    # model = SpringNN(num_subnets=3, subnet_ranges=subnet_ranges, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20001):
        t = torch.rand((500, 1), device=device, requires_grad=True)
        
        loss = loss_fn(model, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    viz(model, device)
