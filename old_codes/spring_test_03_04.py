import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import time

# Constants
M = 1.0  # mass
D = 2
W0 = 80
MU = 2 * M * D  # damping
K = W0 * W0 * M  # spring constant
U0 = 1.0  # initial displacement
V0 = 0.0  # initial velocity

# Utility Functions
# def cosine_window(x, xmin, xmax, device='cpu'):
#     x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1
#     window = (1 + torch.cos(x_scaled * torch.pi)) / 2
#     window = torch.where((x >= xmin) & (x <= xmax), window, torch.tensor(0.0, device=device))
#     return window

def cosine_window(x, mid, width, device='cpu'):
    # x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1
    xmin = mid - width/2
    xmax = mid + width/2
    # x_scaled = 1/sd * x
    x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1
    window = (1 + torch.cos(x_scaled * torch.pi)) / 2
    window = torch.where((x >= xmin) & (x <= xmax), window, torch.tensor(0.0, device=device))
    return window

def norm(mu, sd, x):
    return ((x - mu) / sd).float()

def unnorm(mu, sd, x):
    return (x * sd + mu).float()

# Model Definitions
class SpringSubNN(nn.Module):
    def __init__(self, device, mid, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        ).to(device)
        # self.start = torch.tensor(start, device=device)
        # self.width = torch.tensor(width, device=device)
        # self.end = self.start + self.width
        self.device = device
        self.min_val = mid - width/2
        self.max_val = mid + width/2
        self.mu = (self.min_val+self.max_val)/2
        self.sd = (self.max_val-self.min_val)/2
        self.mid = mid 
        self.width = width
        self.to(device)
        

    # def forward(self, t):
    #     end = self.start + self.width
    #     window = cosine_window(t, self.start, end, self.device)
    #     mu, sd = (self.start + self.width) / 2, self.width / 2
    #     normalized_input = norm(mu, sd, t)
    #     raw_value = self.net(normalized_input)
    #     unnormalized_output = unnorm(mu, sd, raw_value)
    #     return window * unnormalized_output

    # def forward(self, t):
    #     normalized_input = norm(self.mu, self.sd, t)
    #     raw_value = self.net(normalized_input)
    #     unnormalized_output = unnorm(self.mu, self.sd, raw_value)
    #     window = cosine_window(t, self.start, self.end, self.device)
    #     output = window * unnormalized_output
    #     return output


    def forward(self, t):
        normalized_input = norm(self.mu, self.sd, t)
        raw_value = self.net(normalized_input)
        mu = 0
        sd = 1
        unnormalized_output = unnorm(mu, sd, raw_value)
        window = cosine_window(t, self.mid, self.width, self.device)
        output = window * unnormalized_output
        return output, window, unnormalized_output

    def predict(self, t):
        normalized_input = norm(self.mu, self.sd, t)
        raw_value = self.net(normalized_input)
        mu = 0
        sd = 1
        unnormalized_output = unnorm(mu, sd, raw_value)
        window = cosine_window(t, self.mid, self.width, self.device)
        output = window * unnormalized_output
        return output, unnormalized_output

class SpringNN(nn.Module):
    def __init__(self, decomposition_init_kwargs, device):
        super().__init__()
        self.subnets = nn.ModuleList([
            SpringSubNN(device, torch.tensor(mid, device=device), torch.tensor(width, device=device))
            for mid, width in zip(decomposition_init_kwargs['subdomain_xs'], decomposition_init_kwargs['subdomain_ws'])
        ])
        self.device = device

    def forward(self, t):
        outputs = []
        windows = []
        for subnet in self.subnets:
            val, window, raw_val = subnet(t)
            outputs.append(val)
            windows.append(window)
        #TODO: if divide by window
        outputs = torch.stack(outputs, dim=0)
        outputs = torch.sum(outputs, dim=0)
        if True:
            windows = torch.stack(windows, dim=0)
            weight = torch.sum(windows, dim=0)
            outputs = outputs/weight 
        return outputs

    def subnet_outputs(self, t):
        return [subnet.predict(t) for subnet in self.subnets]


def exact_solution(x_batch, device):
    w = torch.sqrt(torch.tensor([W0**2 - D**2], device=device))
    phi = torch.arctan(torch.tensor([-D/w], device=device))
    A = 1 / (2 * torch.cos(phi))
    cos = torch.cos(phi + w * x_batch)
    exp = torch.exp(-D * x_batch)
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
    for i, subnet in enumerate(model.subnets):
        t_vals = torch.linspace(subnet.min_val, subnet.max_val, steps=50, device=device).reshape(-1, 1)
        with torch.no_grad():
            predicted_output, predicted_raws = subnet.predict(t_vals)
        plt.plot(t_vals.cpu().numpy(), predicted_output.cpu().numpy(), label=f'Subnet {i+1} Output', linestyle='--')
    plt.title('Subnet Outputs')
    
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.xlim(0, 1)
    # plt.legend()
    plt.grid(True)
    plt.savefig('spring_windowed_subnets.png')
    plt.close()


    plt.figure(figsize=(10, 6))
    # Plot each subnet's windowed output
    for i, subnet in enumerate(model.subnets):
        t_vals = torch.linspace(subnet.min_val, subnet.max_val, steps=50, device=device).reshape(-1, 1)
        with torch.no_grad():
            predicted_output, predicted_raws = subnet.predict(t_vals)
        plt.plot(t_vals.cpu().numpy(), predicted_raws.cpu().numpy(), label=f'Subnet {i+1} Output', linestyle='--')
    plt.title('Subnet Outputs')
    
    plt.xlabel('Time')
    plt.ylabel('Displacement')
    plt.xlim(0, 1)
    # plt.legend()
    plt.grid(True)
    plt.savefig('spring_raw_subnets.png')
    plt.close()


def plot_subdomain_windows(decomposition_init_kwargs, device='cpu'):
    # Prepare the input range (e.g., 0 to 1)
    t_vals = torch.linspace(0, 1, 1000, device=device)
    
    plt.figure(figsize=(10, 6))
    
    subdomain_xs = torch.linspace(*decomposition_init_kwargs['unnorm'], decomposition_init_kwargs['subdomain_xs'].size)
    subdomain_ws = torch.tensor(decomposition_init_kwargs['subdomain_ws'], device=device)
    
    for i in range(len(subdomain_xs)):
        # start = subdomain_xs[i] - subdomain_ws[i] / 2
        # end = subdomain_xs[i] + subdomain_ws[i] / 2
        
        # # Compute the window function for the current subdomain
        # window_vals = cosine_window(t_vals, start, end, device)

        mid = subdomain_xs[i]
        width = subdomain_ws[i]
        window_vals = cosine_window(t_vals, mid, width, device)
        
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

# FBPINN specific functions
def sample_subdomain_points(model, t):
    # Sample points in the subdomain and calculate corresponding NN outputs and gradients
    # t = 2*torch.rand((200, 1), device=device, requires_grad=True)-1
    # t *= model.sd 
    # t += model.mu
    # normalized_input = norm(model.mu, model.sd, t)
    # raw_value = model.net(normalized_input)
    # unnormalized_output = unnorm(model.mu, model.sd, raw_value)
    # window = cosine_window(t, model.start, model.end, model.device)
    # output = window * unnormalized_output
    output, window, raw_output = model(t)
    #     ones = torch.ones_like(u)
    # u_t = grad(u, t, grad_outputs=ones, create_graph=True)[0]
    # u_tt = grad(u_t, t, grad_outputs=ones, create_graph=True)[0]
    gradient = grad(output, t, grad_outputs=t, create_graph=True)[0]
    gradient2 = grad(gradient, t, grad_outputs=t, create_graph=True)[0]
    return output, gradient, gradient2, window, raw_output

#TODO: fix t issue
def sum_overlapping_regions(outputs, gradients, gradient2s, subnets, ts):
    # Sum NN outputs and gradients in overlapping regions
    for i, subnet in enumerate(subnets):
        start, end = subnet.start, subnet.start + subnet.width
        for j, other_subnet in enumerate(subnets):
            if i != j and end > other_subnet.start and start < other_subnet.start + other_subnet.width:
                output_i, gradient_i, gradient2_i = outputs[i], gradients[i], gradient2s[i]
                output_j, gradient_j, gradient2_j = outputs[j], gradients[j], gradient2s[j]
                # Overlapping window between subnet i and subnet j
                overlap_window = cosine_window(t, max(start, other_subnet.start), min(end, other_subnet.start + other_subnet.width), model.device)
                outputs[i] = outputs[i] + overlap_window * output_j.detach()
                gradients[i] = gradients[i] + overlap_window * gradient_j.detach()
                gradient2s[i] = gradient2s[i] + overlap_window * gradient2_j.detach()
    return outputs, gradients, gradient2s

def sum_all_regions(outputs, gradients, gradient2s, windows):
    stacked_outputs = torch.stack(outputs, dim=0)
    stacked_gradients = torch.stack(gradients, dim=0)
    stacked_gradient2s = torch.stack(gradient2s, dim=0)
    stacked_windows = torch.stack(windows, dim=0)
    
    total_outputs = torch.sum(stacked_outputs, dim=0)
    total_gradients = torch.sum(stacked_gradients, dim=0)
    total_gradient2s = torch.sum(stacked_gradient2s, dim=0)
    total_windows = torch.sum(stacked_windows, dim=0)
    total_outputs /= total_windows
    total_gradients /= total_windows
    total_gradient2s /= total_windows
    
    return total_outputs, total_gradients, total_gradient2s


    

def train_step(model, optimizer, t):
    optimizer.zero_grad()
    # outputs, gradients, gradient2s = [], [], []
    # windows, raw_values = [], []
    # for subnet in model.subnets:
    #     output, gradient, gradient2, window, raw_value = sample_subdomain_points(subnet, t)
    #     outputs.append(output)
    #     gradients.append(gradient)
    #     gradient2s.append(gradient2)
    #     windows.append(window)
    #     raw_values.append(raw_value)
    # outputs, gradients, gradient2s = sum_overlapping_regions(outputs, gradients, gradient2s, model.subnets)
    # outputs, gradients, gradient2s = sum_all_regions(outputs, gradients, gradient2s, windows)
    # Compute loss (including boundary conditions)
    # loss, de_loss, ic_loss_u, ic_loss_v = loss_fn(model, t, outputs, gradients, gradient2s)
    loss, de_loss, ic_loss_u, ic_loss_v = loss_fn_0(model, t)
    loss.backward()
    optimizer.step()
    return loss, de_loss, ic_loss_u, ic_loss_v

def loss_fn_0(model, t):
    u = model(t)
    ones = torch.ones_like(u)
    u_t = grad(u, t, grad_outputs=ones, create_graph=True)[0]
    u_tt = grad(u_t, t, grad_outputs=ones, create_graph=True)[0]
    # u_tt = 0
    
    de_loss = (M * u_tt + MU * u_t + K * u).pow(2).mean()
    
    t0 = torch.tensor([[0.0]], device=model.device, requires_grad=True)
    u0_ = model(t0)
    ut0_ = grad(u0_, t0, create_graph=True)[0]
    
    ic_loss_u = 1e6 * (u0_ - U0).pow(2)
    ic_loss_v = 1e2 * (ut0_ - V0).pow(2)
    
    total_loss = de_loss + ic_loss_u + ic_loss_v
    return total_loss, de_loss, ic_loss_u, ic_loss_v

# def loss_fn(model, t):
#     u = model(t)
#     ones = torch.ones_like(u)
#     u_t = grad(u, t, grad_outputs=ones, create_graph=True)[0]
#     u_tt = grad(u_t, t, grad_outputs=ones, create_graph=True)[0]
#     # u_tt = 0
    
#     de_loss = (m * u_tt + mu * u_t + k * u).pow(2).mean()
    
#     t0 = torch.tensor([[0.0]], device=model.device, requires_grad=True)
#     u0_ = model(t0)
#     ut0_ = grad(u0_, t0, create_graph=True)[0]
    
#     ic_loss_u = 1e6 * (u0_ - u0).pow(2)
#     ic_loss_v = 1e2 * (ut0_ - v0).pow(2)
    
#     return de_loss + ic_loss_u + ic_loss_v
# Loss Function Modification
def loss_fn_2(model, t, outputs, gradients, gradient2s):
    # Calculate the physics-informed part of the loss function
    de_loss = 0.0
    for output, gradient, gradient2 in zip(outputs, gradients, gradient2s):
        u_t = gradient
        # u_tt = grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        u_tt = gradient2
        l = (M * u_tt + MU * u_t + K * output).pow(2)
        # print(l)
        de_loss += l

    de_loss /= len(outputs)
    
    # Apply initial conditions at t=0
    t0 = torch.tensor([[0.0]], device=model.device, requires_grad=True)
    u0_ = model(t0)
    ut0_ = grad(u0_, t0, create_graph=True)[0]
    
    ic_loss_u = 1e6 * (u0_ - U0).pow(2)
    ic_loss_v = 1e2 * (ut0_ - V0).pow(2)
    
    # Sum the differential equation loss with the initial condition losses
    total_loss = de_loss + ic_loss_u + ic_loss_v
    
    return total_loss


def loss_fn(model, t, outputs, gradients, gradient2s):
    # Assuming outputs, gradients, and gradient2s are batched tensors
    # Calculate the physics-informed part of the loss function for the entire batch
    u_t = gradients
    u_tt = gradient2s
    
    # Physics-based loss calculated over the batch
    de_loss = (M * u_tt + MU * u_t + K * outputs).pow(2).mean()
    
    # Apply initial conditions at t=0 for the entire batch
    # Note: The initial condition is the same for all batch items, so we broadcast them
    t0_batch = torch.tensor([[0.0]] * t.size(0), device=model.device, requires_grad=True)
    u0_batch = model(t0_batch)
    ut0_batch = grad(u0_batch.sum(), t0_batch, create_graph=True)[0]
    
    # Calculate initial condition losses for the batch
    # Note: U0 and V0 are scalars, so they broadcast to match the batch size automatically
    ic_loss_u = 1e4*((u0_batch - U0).pow(2)).mean()
    ic_loss_v = 1e2*((ut0_batch - V0).pow(2)).mean()
    
    # Sum the differential equation loss with the initial condition losses
    total_loss = de_loss + ic_loss_u + ic_loss_v
    
    return total_loss, de_loss, ic_loss_u, ic_loss_v



# Main Logic
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Assuming decomposition logic to define subdomains
    decomposition_init_kwargs = {
        'subdomain_xs': np.linspace(0, 1, 15),  # 15 equally spaced subdomains
        'subdomain_ws': 0.15 * np.ones((15,)),  # with widths of 0.15
        'unnorm': (0., 1.),  # define unnormalisation of the subdomain networks
    }

    # decomposition_init_kwargs = {
    #     'subdomain_xs': np.linspace(0, 1, 5),  # 15 equally spaced subdomains
    #     'subdomain_ws': 0.75 * np.ones((5,)),  # with widths of 0.15
    #     'unnorm': (0., 1.),  # define unnormalisation of the subdomain networks
    # }

    # decomposition_init_kwargs = {
    #     'subdomain_xs': np.linspace(0, 1, 1),  # 15 equally spaced subdomains
    #     'subdomain_ws': 3 * np.ones((1,)),  # with widths of 0.15
    #     'unnorm': (0., 1.),  # define unnormalisation of the subdomain networks
    # }
    plot_subdomain_windows(decomposition_init_kwargs, device)
    model = SpringNN(decomposition_init_kwargs, device)
    model.to(device)
    # ... [initialization and plotting code remains the same]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    for epoch in range(20001):
        t = torch.rand((200, 1), device=device, requires_grad=True)
        
        loss, de_loss, ic_loss_u, ic_loss_v = train_step(model, optimizer, t)
        
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            print(f"physics loss:{de_loss}, u0 loss: {ic_loss_u}, v0 loss: {ic_loss_v}" )
            viz(model, device)
            end_time = time.time()
            print(f"time:{end_time-start_time}")
            start_time = end_time
