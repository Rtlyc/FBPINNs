
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import time

# Constants
m = 1.0  # mass
d = 2
w0 = 80
mu = 2 * m * d  # damping
k = w0 * w0 * m  # spring constant
u0 = 1.0  # initial displacement
v0 = 0.0  # initial velocity


def cosine_window(x, xmin, xmax, device='cpu'):
    x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1
    window = (1 + torch.cos(x_scaled * torch.pi)) / 2
    window = torch.where((x >= xmin) & (x <= xmax), window, torch.tensor(0.0, device=device))
    return window

def norm(mu, sd, x):
    return (x-mu)/sd

def unnorm(mu, sd, x):
    return x*sd + mu



class SpringSubNN(nn.Module):
    def __init__(self, device, start, width):
        super(SpringSubNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).to(device)
        self.start = start
        self.width = width
        self.device = device
        # self.mu = 

    def forward(self, t):
        # Apply the window function within the forward method
        end = self.start + self.width
        window = cosine_window(t, self.start, end)
        mu, sd = (self.start+self.width)/2, self.width/2
        normalized_input = norm(mu, sd, t)
        raw_value = self.net(normalized_input)
        unnormalized_output = unnorm(mu, sd, raw_value)
        return window * self.net(t)
    
    # def cosine_window(self, x, xmin, xmax):
    #     x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1
    #     window = (1 + torch.cos(x_scaled * torch.pi)) / 2
    #     window = torch.where((x >= xmin) & (x <= xmax), window, torch.tensor(0.0, device=self.device))
    #     return window


class SpringNN(nn.Module):
    def __init__(self, decomposition_init_kwargs, device):
        super(SpringNN, self).__init__()
        subdomain_xs = torch.linspace(*decomposition_init_kwargs['unnorm'], decomposition_init_kwargs['subdomain_xs'][0].size)
        subdomain_ws = torch.tensor(decomposition_init_kwargs['subdomain_ws'][0], device=device)
        self.subnets = nn.ModuleList()
        for i in range(len(subdomain_xs)):
            start = subdomain_xs[i] - subdomain_ws[i] / 2
            self.subnets.append(SpringSubNN(device, start, subdomain_ws[i]))
        self.device = device

    def forward(self, t):
        return sum(subnet(t) for subnet in self.subnets)
    
    def subnet_outputs(self, t):
        return [subnet(t) for subnet in self.subnets]


def loss_fn(model, t):
    u = model(t)
    ones = torch.ones_like(u)
    u_t = grad(u, t, grad_outputs=ones, create_graph=True)[0]
    u_tt = grad(u_t, t, grad_outputs=ones, create_graph=True)[0]
    # u_tt = 0
    
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
    
    # Get both overall predictions and individual subnet outputs
    with torch.no_grad():
        predictions = model(t_vals)
        predictions = predictions.cpu().numpy()
        subnet_outputs = model.subnet_outputs(t_vals)

    sol = exact_solution(t_vals.squeeze(), device).cpu().numpy()
    
    # Plot overall model predictions and exact solution
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals.cpu().numpy(), predictions, label='NN Prediction')
    plt.plot(t_vals.cpu().numpy(), sol, label='Exact Solution')
    plt.title('Overall Model Predictions vs. Exact Solution')
    plt.grid(True)
    plt.legend()
    plt.savefig('spring_test_overall.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    # Plot each subnet's windowed output
    for i, subnet_output in enumerate(subnet_outputs):
        plt.plot(t_vals.cpu().numpy(), subnet_output.cpu().numpy(), label=f'Subnet {i+1} Output', linestyle='--')
    plt.title('Subnet Outputs')
    
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    # plt.legend()
    plt.grid(True)
    plt.savefig('spring_test.png')
    plt.close()


def plot_subdomain_windows(decomposition_init_kwargs, device='cpu'):
    # Prepare the input range (e.g., 0 to 1)
    t_vals = torch.linspace(0, 1, 1000, device=device)
    
    plt.figure(figsize=(10, 6))
    
    subdomain_xs = torch.linspace(*decomposition_init_kwargs['unnorm'], decomposition_init_kwargs['subdomain_xs'][0].size)
    subdomain_ws = torch.tensor(decomposition_init_kwargs['subdomain_ws'][0], device=device)
    
    for i in range(len(subdomain_xs)):
        start = subdomain_xs[i] - subdomain_ws[i] / 2
        end = subdomain_xs[i] + subdomain_ws[i] / 2
        
        # Compute the window function for the current subdomain
        window_vals = cosine_window(t_vals, start, end, device)
        
        # Plot the window function for the current subdomain
        plt.plot(t_vals.cpu().numpy(), window_vals.cpu().numpy(), label=f'Subdomain {i+1}')
    
    plt.title('Window Functions for Each Subdomain')
    plt.xlabel('Input')
    plt.ylabel('Window Value')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    # plt.show()
    plt.savefig('spring_test_subdomain_windows.png')
    plt.close()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming decomposition logic to define subdomains
    decomposition_init_kwargs = {
        'subdomain_xs': [np.linspace(0, 1, 15)],  # 15 equally spaced subdomains
        'subdomain_ws': [0.15 * np.ones((15,))],  # with widths of 0.15
        'unnorm': (0., 1.),  # define unnormalisation of the subdomain networks
    }
    plot_subdomain_windows(decomposition_init_kwargs, device)
    model = SpringNN(decomposition_init_kwargs, device)
    model.to(device)

    # model = SpringNN(device, **decomposition_init_kwargs)
    # subnet_ranges = [(0, 0.4), (0.3, 0.7), (0.6, 1.0)]
    # model = SpringNN(num_subnets=3, subnet_ranges=subnet_ranges, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()
    for epoch in range(20001):
        t = torch.rand((200, 1), device=device, requires_grad=True)
        
        loss = loss_fn(model, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            viz(model, device)
            end_time = time.time()
            print(f"time:{end_time-start_time}")
            start_time = end_time