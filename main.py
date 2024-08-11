""" 
This script serves as an executing script to transform the original model into a submodel manner. To do this, we need to split the original model into parts: 1. encoder 2. window function 3. symmetric operator 
"""

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import os 
from datetime import datetime, timedelta
import numpy as np
import pickle
import yaml
import random
import math 
from torch.nn import Linear
from torch.autograd import Variable
from torch import Tensor
from models import NN_0624 as NN
from torch.utils.tensorboard import SummaryWriter
import time
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity

import gridmap

USE_FILTER = True

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class ConditionalProfiler:
    def __init__(self, enabled=True, profile_dir='./log_dir', max_steps=None):
        self.enabled = enabled
        self.profile_dir = profile_dir
        self.max_steps = max_steps
        self.step_count = 0
        self.profiler = None

    def __enter__(self):
        if self.enabled:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.__enter__()  # Start profiling
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.profiler:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
            # self.profiler.export_chrome_trace(f"{self.profile_dir}/trace.json")

    def step(self):
        if self.enabled and self.profiler:
            self.profiler.step()
            self.step_count += 1
            if self.max_steps and self.step_count >= self.max_steps:
                self.enabled = False  # Stop profiling after max_steps
                self.profiler.__exit__(None, None, None)  # Cleanly exit profiler


def cosine_2d(xmin, xmax, ymin, ymax, x, y):
    mu_x, sd_x = (xmin + xmax) / 2, (xmax - xmin) / 2
    mu_y, sd_y = (ymin + ymax) / 2, (ymax - ymin) / 2
    
    w_x = ((1 + torch.cos(torch.pi * (x - mu_x) / sd_x)) / 2) ** 2
    w_y = ((1 + torch.cos(torch.pi * (y - mu_y) / sd_y)) / 2) ** 2
    
    w_x = torch.where((x>=xmin) & (x<=xmax), w_x, torch.tensor(0.001, device=x.device))
    w_y = torch.where((y>=ymin) & (y<=ymax), w_y, torch.tensor(0.001, device=y.device))

    return w_x * w_y

def gaussian_2d(xmin, xmax, ymin, ymax, x, y):
    a = 1
    mu_x, sd_x = (xmin + xmax) / 2, (xmax - xmin) / 2 / 3
    mu_y, sd_y = (ymin + ymax) / 2, (ymax - ymin) / 2 / 3

    w_x = a * torch.exp(-((x - mu_x) ** 2) / (2 * sd_x ** 2))
    w_y = a * torch.exp(-((y - mu_y) ** 2) / (2 * sd_y ** 2))

    w_x = torch.where((x>=xmin) & (x<=xmax), w_x, torch.tensor(0.00, device=x.device))
    w_y = torch.where((y>=ymin) & (y<=ymax), w_y, torch.tensor(0.00, device=y.device))
    return w_x * w_y

def data_filter_by_regions(frame_data, regions):
    valid_start_indices = torch.zeros(frame_data.size(0), dtype=torch.bool, device=frame_data.device)
    valid_end_indices = torch.zeros(frame_data.size(0), dtype=torch.bool, device=frame_data.device)
    for region in regions:
        xmin, xmax, ymin, ymax = region
        valid_start_indices |= (frame_data[:, 0] >= xmin) & (frame_data[:, 0] <= xmax) & (frame_data[:, 1] >= ymin) & (frame_data[:, 1] <= ymax)
        valid_end_indices |= (frame_data[:, 3] >= xmin) & (frame_data[:, 3] <= xmax) & (frame_data[:, 4] >= ymin) & (frame_data[:, 4] <= ymax)
    valid_indices = valid_start_indices & valid_end_indices
    return frame_data[valid_indices]

# def scale_region(region, coord):
#     # scale the region to -0.5, 0.5
#     xmin, xmax, ymin, ymax = region
#     kx = (xmax - xmin)

def plot_window_2d_normalized(regions, savepath, columns=3, rows=3):
    # xmin, xmax, ymin, ymax = -0.5, 0.5, -0.5, 0.5
    # xmin, xmax, ymin, ymax = 
    regions = np.array(regions)
    xmin, xmax, ymin, ymax = regions[:, 0].min(), regions[:, 1].max(), regions[:, 2].min(), regions[:, 3].max()
    n = 100  # resolution of the grid
    x = torch.linspace(xmin, xmax, n)
    y = torch.linspace(ymin, ymax, n)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    # Compute weights for all regions and store them
    all_weights = []
    for region in regions:
        xmin, xmax, ymin, ymax = region
        # weights = cosine_2d(torch.tensor(xmin), torch.tensor(xmax), torch.tensor(ymin), torch.tensor(ymax), xx, yy)
        weights = gaussian_2d(torch.tensor(xmin), torch.tensor(xmax), torch.tensor(ymin), torch.tensor(ymax), xx, yy)
        all_weights.append(weights)

    # Calculate the sum of all weights
    total_weights = torch.stack(all_weights).sum(dim=0)

    # plt.figure(figsize=(12, 10))
    plt.figure(figsize=(rows*2, columns*2))
    for i, region in enumerate(regions, 1):
        # Normalize the weights for this region
        normalized_weights = all_weights[i - 1] / total_weights
        # plt.subplot(int(math.sqrt(len(regions))), int(math.sqrt(len(regions))), i)
        plt.subplot(rows, columns, i)
        cp = plt.contourf(xx.numpy(), yy.numpy(), normalized_weights.numpy(), levels=50, cmap='viridis', vmin=0, vmax=1)
        # plt.colorbar(cp, ticks=[0, 1])
        xmin, xmax, ymin, ymax = region
        # plt.title(f'Region {i}: [{xmin:.3f},{xmax:.3f}] x [{ymin:.3f},{ymax:.3f}]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(savepath + "/window_2d_plot_normalized.png")
    plt.close()


"""
SubModel is a subnet of Model. In our example, we want to maintain 4 subnets in Model. Each subnet should encode the input the start and goal depending on some window function. The output of the subnet should be the encoded features(including start and goal) after the window function. But the real output of the model should be the normalized features from all the subnets. Thus, we need to output the weights of the subnets. 
"""
class SubModel(torch.nn.Module):
    def __init__(self, region, config, device):
        super(SubModel, self).__init__()
        self.config = config
        self.dim = config["data"]["dim"]
        self.device = device 
        self.region = region 
        self.layer_sizes = config["model"]["layer_sizes"]
        self.B = 0.5*torch.normal(0,1, size=(128,self.dim), device=self.device).T
        # self.b = 10*torch.rand(1, 128)

        self.nl = 4
        self.input_size = self.B.shape[1]
        self.h_size = 64

        self.init_network()
        self.apply(self.init_weights)
        self.to(self.device)

    def init_network(self):
        self.encoder = NN.SubNN.init_network(self.layer_sizes, self.input_size)
        self.norm_layers = nn.ModuleList([nn.InstanceNorm1d(size) for size in self.layer_sizes])

    def input_mapping(self, x):
        w = 2.*np.pi*self.B
        x_proj = x @ w
        #x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear:
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            #stdv = np.sqrt(1 / 32.) / 4
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)
    
    def encoder_out_new(self, coords):
        if self.config['model']['normalization']:
            #!: scale the input to -0.5, 0.5 (only for x and y)
            #* (x-xmin)*(1/(xmax-xmin)) - 0.5 = x*(1/(xmax-xmin)) - xmin*(1/(xmax-xmin)) - 0.5
            x_scale = 1 / (self.region[1] - self.region[0])
            y_scale = 1 / (self.region[3] - self.region[2])

            # Create a scaling matrix
            scaling_matrix = torch.eye(coords.size(1), device=self.device)  # Identity matrix of appropriate size
            scaling_matrix[0, 0] = x_scale
            scaling_matrix[1, 1] = y_scale
            # scaling_matrix[3, 3] = x_scale
            # scaling_matrix[4, 4] = y_scale

            # Create offset vectors
            offsets = torch.zeros(coords.size(1), device=self.device)
            offsets[0] = -self.region[0] * x_scale - 0.5
            offsets[1] = -self.region[2] * y_scale - 0.5
            # offsets[3] = -self.region[0] * x_scale - 0.5
            # offsets[4] = -self.region[2] * y_scale - 0.5

            # Perform the dot product and add the offsets
            scaled_coords = torch.matmul(coords, scaling_matrix) + offsets
            coords = scaled_coords
        return NN.SubNN.encoder_out_new(self, coords)

    def out_new(self, coords):
        # ? this should only output encoder features
        size = coords.shape[0]
        encoder_output, _ = self.encoder_out_new(coords)
        # x0 = coords[:,:self.dim]
        # x1 = coords[:,self.dim:]
        # x_stack = torch.vstack((x0, x1))
        # window = cosine_2d(*self.region, x_stack[:, 0, None], x_stack[:, 1, None])
        # window = gaussian_2d(*self.region, x_stack[:, 0, None], x_stack[:, 1, None])
        window = gaussian_2d(*self.region, coords[:, 0], coords[:, 1])
        return encoder_output, window


    def encoder_out(self, coords):
        if not USE_FILTER:
            if self.config['model']['normalization']:
                #!: scale the input to -0.5, 0.5 (only for x and y)
                #* (x-xmin)*(1/(xmax-xmin)) - 0.5 = x*(1/(xmax-xmin)) - xmin*(1/(xmax-xmin)) - 0.5
                x_scale = 1 / (self.region[1] - self.region[0])
                y_scale = 1 / (self.region[3] - self.region[2])

                # Create a scaling matrix
                scaling_matrix = torch.eye(coords.size(1), device=self.device)  # Identity matrix of appropriate size
                scaling_matrix[0, 0] = x_scale
                scaling_matrix[1, 1] = y_scale
                scaling_matrix[3, 3] = x_scale
                scaling_matrix[4, 4] = y_scale

                # Create offset vectors
                offsets = torch.zeros(coords.size(1), device=self.device)
                offsets[0] = -self.region[0] * x_scale - 0.5
                offsets[1] = -self.region[2] * y_scale - 0.5
                offsets[3] = -self.region[0] * x_scale - 0.5
                offsets[4] = -self.region[2] * y_scale - 0.5

                # Perform the dot product and add the offsets
                scaled_coords = torch.matmul(coords, scaling_matrix) + offsets
                coords = scaled_coords
        else:
            if self.config['model']['normalization']:
                #!: scale the input to -0.5, 0.5 (only for x and y)
                #* (x-xmin)*(1/(xmax-xmin)) - 0.5 = x*(1/(xmax-xmin)) - xmin*(1/(xmax-xmin)) - 0.5
                x_scale = 1 / (self.region[1] - self.region[0])
                y_scale = 1 / (self.region[3] - self.region[2])

                # Create a scaling matrix
                scaling_matrix = torch.eye(coords.size(1), device=self.device)  # Identity matrix of appropriate size
                scaling_matrix[0, 0] = x_scale
                scaling_matrix[1, 1] = y_scale
                # scaling_matrix[3, 3] = x_scale
                # scaling_matrix[4, 4] = y_scale

                # Create offset vectors
                offsets = torch.zeros(coords.size(1), device=self.device)
                offsets[0] = -self.region[0] * x_scale - 0.5
                offsets[1] = -self.region[2] * y_scale - 0.5
                # offsets[3] = -self.region[0] * x_scale - 0.5
                # offsets[4] = -self.region[2] * y_scale - 0.5

                # Perform the dot product and add the offsets
                scaled_coords = torch.matmul(coords, scaling_matrix) + offsets
                coords = scaled_coords
        
        return NN.SubNN.encoder_out(self, coords)
    

    def out(self, coords):
        if not USE_FILTER:
            # ? this should only output encoder features
            size = coords.shape[0]
            encoder_output, _ = self.encoder_out(coords)
            x0 = coords[:,:self.dim]
            x1 = coords[:,self.dim:]
            x_stack = torch.vstack((x0, x1))
            # window = cosine_2d(*self.region, x_stack[:, 0, None], x_stack[:, 1, None])
            window = gaussian_2d(*self.region, x_stack[:, 0, None], x_stack[:, 1, None])
        else:
            encoder_output, _ = self.encoder_out(coords)
            window = gaussian_2d(*self.region, coords[:, 0], coords[:, 1])
        return encoder_output, window

""" 
Model maintains the most original model and feature. We should only care about the new output is composed of the normalized features from all the subnets.
"""
class Model(torch.nn.Module):
    def __init__(self, ModelPath, config_path, device="cpu"):
        super(Model, self).__init__()
        self.Params = {}
        self.Params['ModelPath'] = ModelPath

        # self.scale_factor = scale_factor
        current_time = datetime.utcnow()-timedelta(hours=5)
        self.folder = self.Params['ModelPath']+"/"+current_time.strftime("%m_%d_%H_%M")
        self.writer = SummaryWriter(self.folder)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)


        # Pass the JSON information
        self.Params['Device'] = device
        self.Params['Pytorch Amp (bool)'] = False

        self.Params['Network'] = {}
        self.Params['Network']['Normlisation'] = 'OffsetMinMax'

        self.Params['Training'] = {}
        self.Params['Training']['Number of sample points'] = 2e5
        self.Params['Training']['Batch Size'] = 2000
        self.Params['Training']['Validation Percentage'] = 10
        self.Params['Training']['Number of Epochs'] = 20000
        self.Params['Training']['Resampling Bounds'] = [0.1, 0.9]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 50
        self.Params['Training']['Learning Rate'] = 1e-3#5e-5
        self.Params['Training']['Random Distance Sampling'] = True
        self.Params['Training']['Use Scheduler (bool)'] = False

        # Parameters to alter during training
        self.total_train_loss = []
        self.total_val_loss = []
        self.epoch = 0
        self.frame_idx = 0
        self.trajectory = None
        self.prev_state_queue = []
        self.prev_optimizer_queue = []
        self.timer = []
        self.frame_buffer_size = 20
        self.camera_steps = 5000//50
        self.minimum = 0.01 #0.02
        self.maximum = 0.1  #0.1
        self.all_framedata = None
        self.all_surf_pc = []
        self.free_pc = []
        self.device = device


        # self.initial_view = Tensor([-0.3, -0.2, 0])
        self.initial_view = Tensor([-1.5, -1.0, 0])
        self.initial_view = Tensor([-0.0, -0.0, 0])
        self.initial_view = Tensor([-2.0, -0.0, 0])
        self.initial_view = Tensor([-1.5, 1.0, 0])


                #* Load the configuration file
        with open(config_path, 'r') as file:
           self.config = yaml.safe_load(file)
        self.dim = self.config["data"]["dim"]
        self.scale_factor = self.config["data"]["scaling"]
        self.initial_view = Tensor(self.config["model"]["initial_view"])

        self.name = self.config["paths"]["name"]
        self.region_combination = self.config['region']["region_combination"]

        self.batch_size = self.config['model']['batch_size']

        self.load_rawdata()
        self.init_network()
        print()

    def init_network(self):
        #! initialize the network, notice that we need to split the original model into submodels

        #! each region looks like
        #! |overlap|******core******|overlap|

        #? 2D window
        region_predefined = self.config["region"]["use_predefined"]
        columns, rows = 3, 3
        
        if region_predefined:
            self.regions = self.config["region"]['regions']
        else:
            self.regions = []
            xmin, xmax, ymin, ymax = self.config["region"]["boundaries"]
            columns = self.config["region"]["columns"]
            rows = self.config["region"]["rows"]
            overlap_ratio = self.config["region"]["overlap_ratio"]

            width_total = xmax - xmin
            height_total = ymax - ymin
            
            width_core = width_total / (columns)
            width_overlap = width_core * overlap_ratio
            column_regions = [(xmin - width_overlap + i*width_core, xmin + width_core+width_overlap + i*width_core) for i in range(columns)]

            height_core = height_total / (rows)
            height_overlap = height_core * overlap_ratio
            row_regions = [(ymin - height_overlap + i*height_core, ymin + height_overlap+height_core + i*height_core) for i in range(rows)]

            # for i in range(rows):
            #     for j in range(columns):
            #         region = (column_regions[j][0], column_regions[j][1], row_regions[i][0], row_regions[i][1])
            #         self.regions.append(region)
            self.block_idx_to_subnet_idx = 1000*torch.ones((columns, rows), dtype=torch.int32)
            for i, j in self.occ_grid.valid_blocks_indices:
                region = (column_regions[i][0], column_regions[i][1], row_regions[j][0], row_regions[j][1])
                self.regions.append(region)
                self.block_idx_to_subnet_idx[i, j] = len(self.regions) - 1



        plot_window_2d_normalized(self.regions, self.folder, columns, rows)
        
        self.subnets = torch.nn.ModuleList([SubModel(region, self.config, self.device) for region in self.regions])
        self.subnets = nn.ModuleList([nn.DataParallel(subnet) for subnet in self.subnets])  # Wrapping in DataParallel

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.Params['Training']['Learning Rate'], weight_decay=0.2)

        self.all_regions = list(range(len(self.regions)))
        self.active_regions = self.all_regions
        self.learned_regions = set()

    def load_rawdata(self):
        if False:
            self.explored_data = np.load("data/explored_data.npy")
            #! hardcode room size
            # self.explored_data[:, :, :6] *= 5.0
            # print("hardcoded room size scale: 5.0")
            points = np.load("data/mesh_points_0.npy")
            speeds = np.load("data/mesh_speeds_0.npy")
            self.explored_data = np.concatenate((points, speeds), axis=1)
        elif True:
            # points = np.load("data/cube_passage_points.npy")
            # speeds = np.load("data/cube_passage_speeds.npy")
            # points = np.load("data/cabin_points.npy")
            # speeds = np.load("data/cabin_speeds.npy")
            points = np.load("data/narrow_cube_points_0.npy").astype(np.float32)
            speeds = np.load("data/narrow_cube_speeds_0.npy").astype(np.float32)
            self.all_bounds = np.load("data/narrow_cube_bounds_0.npy").astype(np.float32)
            # invalid_indices_0 = (points[:, 0] > -1.0) & (points[:, 0] < 1.0) | (points[:, 1] > 0.0)
            # invalid_indices_1 = (points[:, 3] > -1.0) & (points[:, 3] < 1.0) | (points[:, 4] > 0.0)
            # invalid_indices = np.logical_or(invalid_indices_0, invalid_indices_1)

            # points = points[~invalid_indices]
            # speeds = speeds[~invalid_indices] 
            self.explored_data = np.concatenate((points, speeds), axis=1)
            self.explored_data = torch.tensor(self.explored_data, device=self.device)
            self.all_bounds = torch.tensor(self.all_bounds, device=self.device)
        if False:
            allpoints = []
            allspeeds = []
            allbounds = []
            for i in range(len(self.all_regions)):
                points = np.load(f"data/{self.name}_points_{i}.npy").astype(np.float32)
                speeds = np.load(f"data/{self.name}_speeds_{i}.npy").astype(np.float32)
                bounds = np.load(f"data/{self.name}_bounds_{i}.npy").astype(np.float32)
                allpoints.append(points)
                allspeeds.append(speeds)
                allbounds.append(bounds)
            allpoints = np.stack(allpoints)
            allspeeds = np.stack(allspeeds)
            allbounds = np.stack(allbounds)
            self.explored_data = np.concatenate((allpoints, allspeeds), axis=2)
            self.explored_data = torch.tensor(self.explored_data, device=self.device)
            self.all_bounds = torch.tensor(allbounds, device=self.device)
        if False: #b1 dataset
            from data_processing import lidar_sample
            root_dir = '/home/exx/Documents/FBPINNs/b1/haas_fine'
            config_file = "/home/exx/Documents/FBPINNs/configs/lidar.json"
            dataset = lidar_sample.LidarDataset(root_dir, config_file)
            points, speeds, bounds = dataset.get_speeds(range(133), num=500000)
            self.explored_data = torch.cat((points, speeds), dim=1)[None,]
        print("explored data shape:", self.explored_data.shape)

        #! Parameters for the occupancy grid
        region_bounds = self.config["region"]["boundaries"]
        length = max(region_bounds[1] - region_bounds[0], region_bounds[3] - region_bounds[2])
        dim_cells = self.config['occ_map']['dim_cells']
        occupancy_threshold = self.config["occ_map"]["occ_threshold"]
        block_columns = self.config["region"]["columns"]
        block_rows = self.config["region"]["rows"]

        spacing = length / dim_cells
        offset = length / 2
        self.occ_grid = gridmap.OccupancyGridMap(np.ones((dim_cells, dim_cells)), cell_size = spacing, occupancy_threshold=occupancy_threshold, offset=offset, block_cols=block_columns, block_rows=block_rows)


        end_points = self.explored_data.reshape(-1, 8)[:, 3:5]
        end_bounds = self.all_bounds.reshape(-1, 2)[:, 1].unsqueeze(1)
        self.occ_grid.update(self.initial_view.numpy(), end_points, end_bounds)
        self.occ_grid.plot(self.initial_view, end_points, path=self.folder + "/occ_grid_initial.png")
        print()

    def set_requires_grad(self, regions, requires_grad):
        for region in regions:
            subnet = self.subnets[region]
            for param in subnet.parameters():
                param.requires_grad = requires_grad

    def randomize_data_prev(self, frame_data):
        prev_frame_idx = torch.randperm(self.frame_idx).tolist()[:self.frame_buffer_size-1]
        prev_framedata = self.all_framedata[prev_frame_idx]
        chosen_framedata = torch.cat((prev_framedata, frame_data.unsqueeze(0)), dim=0).view(-1, 8)

        chosen_points = chosen_framedata[:,:6]
        chosen_speeds = chosen_framedata[:,6:]
        
        points = frame_data[:,:6]
        speeds = frame_data[:,6:]

        local_points_combination = torch.cat((points[:,:3], points[:,3:]), dim=1)
        local_speeds_combination = torch.cat((speeds[:,:1], speeds[:,1:]), dim=1)

        chosen_percent = 1.0
        # print("chosen points shape:", chosen_points.shape)
        chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        # print("chosen points indice shape:", chosen_start_index.shape)
        chosen_start_points = chosen_points[chosen_start_index][:, :3]
        chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
        chosen_end_points = chosen_points[chosen_end_index][:, 3:6]
        chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]
        # print("chosen start points shape:", chosen_start_points.shape)
        global_points_combination = torch.cat((chosen_start_points, chosen_end_points), dim=1)
        global_speeds_combination = torch.cat((chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)

        all_points = torch.cat((local_points_combination, global_points_combination), dim=0)
        all_speeds = torch.cat((local_speeds_combination, global_speeds_combination), dim=0)

        cur_data = torch.cat((all_points, all_speeds), dim=1)
        return cur_data

    def randomize_data_cur(self, frame_data):
        #! current frame data + random frame_data
        rand_start_idx = torch.randperm(frame_data.shape[0])[:500000]
        rand_end_idx = torch.randperm(frame_data.shape[0])[:500000]
        rand_start_xyz = frame_data[rand_start_idx][:, :3]
        rand_end_xyz = frame_data[rand_end_idx][:, 3:6]
        rand_start_speed = frame_data[rand_start_idx][:, 6:7]
        rand_end_speed = frame_data[rand_end_idx][:, 7:8]
        global_points_combination = torch.cat((rand_start_xyz, rand_end_xyz), dim=1)
        global_speeds_combination = torch.cat((rand_start_speed, rand_end_speed), dim=1)
        global_data = torch.cat((global_points_combination, global_speeds_combination), dim=1)
        # cur_data = global_data
        cur_data = torch.cat((frame_data, global_data), dim=0)
        return cur_data

    def randomize_region_data_prev(self, frame_data):
        #! current frame data + previous frame data(current start and previous end) + random frame_data
        prev_frame_idx = torch.randperm(self.all_framedata.shape[0])[:frame_data.shape[0]]
        prev_frame = self.all_framedata[prev_frame_idx]
        prev_frame_points = torch.cat((frame_data[:, :3], prev_frame[:, 3:6]), dim=1)
        prev_frame_speeds = torch.cat((frame_data[:, 6:7], prev_frame[:, 7:]), dim=1)
        prev_frame_data = torch.cat((prev_frame_points, prev_frame_speeds), dim=1)

        rand_start_idx = torch.randperm(frame_data.shape[0])
        rand_end_idx = torch.randperm(frame_data.shape[0])
        rand_start_xyz = frame_data[rand_start_idx][:, :3]
        rand_end_xyz = frame_data[rand_end_idx][:, 3:6]
        rand_start_speed = frame_data[rand_start_idx][:, 6:7]
        rand_end_speed = frame_data[rand_end_idx][:, 7:8]
        rand_frame_points = torch.cat((rand_start_xyz, rand_end_xyz), dim=1)
        rand_frame_speeds = torch.cat((rand_start_speed, rand_end_speed), dim=1)
        rand_frame_data = torch.cat((rand_frame_points, rand_frame_speeds), dim=1)

        cur_data = torch.cat((frame_data, prev_frame_data, rand_frame_data), dim=0)

        return cur_data

    def region_encode_out(self, x, active_regions):
        # x = Tensor([[-2.5, -2.0, 0, -2.0, -2.0, 0], [-2.5, -2.0, 0, -2.0, -2.0, 0]]).to(self.device)
        if not USE_FILTER:
            #! only normalize active regions
            x = x.clone().detach().requires_grad_(True)
            s_time = time.time()
            feature_array, weight_array = [], []
            for subnet_ind in active_regions:
                subnet = self.subnets[subnet_ind]
                feature, weight = subnet.module.out(x)
                feature_array.append(feature*weight)
                weight_array.append(weight)
            feature_array = torch.stack(feature_array)
            weight_array = torch.stack(weight_array)
            outputs = torch.sum(feature_array, dim=0)
            ws = torch.sum(weight_array, dim=0)
            normalized_features = outputs/ws
            # print("region encode out time origin:", time.time()-s_time)
        if USE_FILTER and True:
            x = x.clone().detach().requires_grad_(True)
            s_time = time.time()
            subnet_valid_indices = []
            x_s = x[:, :3]
            x_e = x[:, 3:]
            x2 = torch.vstack((x_s, x_e))
            region_points_mask = self.occ_grid.get_region_points_mask(x2[:, :2], self.occ_grid.valid_blocks_indices)
            # region_points_mask = self.occ_grid.get_region_points_mask(end)
            features = torch.zeros((x2.shape[0], 256), device=x.device)
            weights = torch.zeros((x2.shape[0], 1), device=x.device)
            for subnet_ind in active_regions:
                subnet = self.subnets[subnet_ind]
                #?: should filter the data and only use the data in the region
                valid_indices = region_points_mask[subnet_ind]
                subnet_valid_indices.append(valid_indices)
                x3 = x2[valid_indices]
                feature, weight = subnet.module.out_new(x3)
                features[valid_indices] += feature*(weight.unsqueeze(1)) 
                weights[valid_indices] += weight.unsqueeze(1)
                # features.append(feature*weight)
                # weights.append(weight)
            # features = torch.stack(features)
            # weights = torch.stack(weights)
            # outputs = torch.sum(features, dim=0)
            normalized_features = features/weights 
            # print("region encode out time new:", time.time()-s_time)
        return normalized_features, x 

    # def region_encode_out_new(self, x, active_regions):
    #     #! only normalize active regions
    #     x = x.clone().detach().requires_grad_(True)
    #     features, weights = [], []
    #     subnet_valid_indices = []
    #     x2 = x.reshape(-1, 3)
    #     region_points_mask = self.occ_grid.get_region_points_mask(x2[:, :2], self.occ_grid.valid_blocks_indices)
    #     # region_points_mask = self.occ_grid.get_region_points_mask(end)
    #     features = torch.zeros((x2.shape[0], 256), device=x.device)
    #     weights = torch.zeros((x2.shape[0], 1), device=x.device)
    #     for subnet_ind in active_regions:
    #         subnet = self.subnets[subnet_ind]
    #         #TODO: should filter the data and only use the data in the region
    #         valid_indices = region_points_mask[subnet_ind]
    #         subnet_valid_indices.append(valid_indices)
    #         x3 = x2[valid_indices]
    #         feature, weight = subnet.module.out(x3)
    #         features[valid_indices] += feature*(weight.unsqueeze(1)) 
    #         weights[valid_indices] += weight.unsqueeze(1)
    #         # features.append(feature*weight)
    #         # weights.append(weight)
    #     # features = torch.stack(features)
    #     # weights = torch.stack(weights)
    #     # outputs = torch.sum(features, dim=0)
    #     normalized_features = features/weights 
    #     # ws = torch.sum(weights, dim=0)
    #     # normalized_features = outputs/ws
    #     return normalized_features, x 
    
    def sym_op(self, normalized_features, Xp):
        return NN.NN.sym_op(self, normalized_features, Xp)

    def out(self, x, regions):
        #! core function to output the normalized features from all the subnets
        normalized_features, Xp = self.region_encode_out(x, regions)
        outputs, Xp = self.sym_op(normalized_features, Xp)
        return outputs, Xp
        
    def gradient(self, y, x, create_graph=True):
        grad_y = torch.ones_like(y)
        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x

    def Loss(self, points, Yobs, beta):
        return NN.NN.Loss(self, points, Yobs, beta)

    def train(self):

        frame_epoch = self.config['model']['frame_epoch']
        self.alpha = 1.0

        region_combination = self.region_combination
        while True:
            if False:
                #? by frame sequence
                frame_data = self.explored_data[self.frame_idx]
                frame_data = torch.tensor(frame_data).to(self.Params['Device'])
            elif True:
                #? by random 
                explored_data = self.explored_data.reshape(-1, 8)
                rand_idx = torch.randperm(explored_data.shape[0])[:1000000]
                frame_data = torch.tensor(explored_data[rand_idx]).to(self.Params['Device'])
                # frame_data[:, :6] *= 0.25
                # frame_data[:, 2] *= 0.5
                # frame_data[:, 5] *= 0.5
                #! filter the data
                # frame_data = data_filter_by_regions(frame_data, self.regions)
            if False:
                #? by region combination
                self.active_regions = region_combination[self.frame_idx % len(region_combination)]
                explored_data = self.explored_data[self.active_regions].reshape(-1, 8)
                # rand_start_idx = torch.randperm(explored_data.shape[0])[:500000]
                # rand_end_idx = torch.randperm(explored_data.shape[0])[:500000]
                # rand_start_xyz = explored_data[rand_start_idx][:, :3]
                # rand_end_xyz = explored_data[rand_end_idx][:, 3:6]
                # rand_start_speed = explored_data[rand_start_idx][:, 6:7]
                # rand_end_speed = explored_data[rand_end_idx][:, 7:8]
                # frame_data = np.concatenate((rand_start_xyz, rand_end_xyz, rand_start_speed, rand_end_speed), axis=1)
                # frame_data = torch.tensor(frame_data).to(self.Params['Device'])

                rand_idx = torch.randperm(explored_data.shape[0])[:500000]
                frame_data = torch.tensor(explored_data[rand_idx]).to(self.Params['Device'])
                self.set_requires_grad(self.active_regions, True)
                self.set_requires_grad(list(set(self.all_regions)-set(self.active_regions)), False)
                # self.set_requires_grad(self.all_regions, True)
            total_diff = self.train_core(frame_epoch, frame_data)

            # self.plot(self.initial_view, self.epoch, total_diff.item(), self.alpha, cur_data[:, :6].clone().cpu().numpy(), None)

            with torch.no_grad():
                self.save(epoch=self.epoch, val_loss=total_diff)

            self.learned_regions.update(self.active_regions)
            # self.learned_regions = set([self.active_regions[-1]])
            # self.learned_regions = set(self.all_regions)

            self.frame_idx += 1
            if self.frame_idx >= len(region_combination):
                break 
            # if self.frame_idx >= len(self.explored_data):
            #     break

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.subnets.state_dict(),
                    'B_state_dicts': [subnet.module.B for subnet in self.subnets],
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.folder, str(epoch).zfill(5), val_loss))
        
    def load(self, filepath):
        #B = torch.load(self.Params['ModelPath']+'/B.pt')
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        Bs = checkpoint['B_state_dicts']

        self.subnets = torch.nn.ModuleList([SubModel(region, self.config, self.device) for region in self.regions])
        self.subnets = nn.ModuleList([nn.DataParallel(subnet) for subnet in self.subnets])  # Wrapping in DataParallel
        for i, subnet in enumerate(self.subnets):
            subnet.module.B = Bs[i]

        self.subnets.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.subnets.to(torch.device(self.device))
        self.subnets.float()
        self.subnets.eval()

    def train_core(self, epoch, frame_data):
        beta = 1.0
        prev_diff = 1.0
        current_diff = 10.0
        step = -1000.0/4000.0
        # if self.all_framedata is None:
        #     self.all_framedata = frame_data.unsqueeze(0)
        # else:
        #     self.all_framedata = torch.cat((self.all_framedata, frame_data.unsqueeze(0)), dim=0)

        if self.all_framedata is None:
            self.all_framedata = frame_data
        else:
            self.all_framedata = torch.cat((self.all_framedata, frame_data), dim=0)
        # print(self.all_framedata.shape)


        #! mix data so that the start and end points are from different frames
        # if is_one_frame:
        # cur_data = self.randomize_data_prev(frame_data)
        # cur_data = frame_data
        # cur_data = self.randomize_data_cur(frame_data)
        cur_data = self.randomize_region_data_prev(frame_data)
        rand_idx = torch.randperm(cur_data.shape[0])[:self.batch_size]
        cur_data = cur_data[rand_idx]
        dataloader = FastTensorDataLoader(cur_data, batch_size=int(self.Params['Training']['Batch Size'])*len(self.regions), shuffle=True)

        frame_epoch = epoch

        #! Profiling
        # # Define the number of steps to profile
        # profile_steps = 10  # Number of batches to profile

        # with torch.profiler.profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=profile_steps, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.folder),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True
        # ) as prof:

        profile_enabled = False
        profile_steps = 5 
        with ConditionalProfiler(profile_enabled, self.folder, profile_steps) as prof:    
            for e in range(frame_epoch):
                epoch_start_time = time.time()
                self.epoch += 1
                total_train_loss = 0
                total_diff=0

                # gamma=0.001#max((4000.0-epoch)/4000.0/20,0.001)
                # mu = 10

                current_state = pickle.loads(pickle.dumps(self.state_dict()))
                current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
                self.prev_state_queue.append(current_state)
                self.prev_optimizer_queue.append(current_optimizer)
                if(len(self.prev_state_queue)>5):
                    self.prev_state_queue.pop(0)
                    self.prev_optimizer_queue.pop(0)
                
                self.optimizer.param_groups[0]['lr']  = 5e-4


                prev_diff = current_diff
                iter=0
                while True:
                    total_train_loss = 0
                    total_diff = 0
                    #for i in range(10):
                    for i, data in enumerate(dataloader, 0):#train_loader_wei,dataloader
                        #print('----------------- Epoch {} - Batch {} --------------------'.format(epoch,i))
                        if i>5:
                            break
        
                        data=data[0].to(self.Params['Device'])
                        #data, indexbatch = data
                        points = data[:,:2*self.dim]#.float()#.cuda()
                        speed = data[:,2*self.dim:]#.float()#.cuda()
                        
                        speed = speed*speed*(2-speed)*(2-speed)
                        speed = self.alpha*speed+1-self.alpha


                        # loss_value, loss_n, wv = self.Loss(
                        # points, speed, beta)
                        # loss_value.backward()
                        with record_function("model_inference"):
                            loss_value, loss_n, wv = self.Loss(
                        points, speed, beta)

                        with record_function("backward_pass"):
                            s = time.time()
                            loss_value.backward()
                            e = time.time()
                            # print('Backward Time: ', e-s)

                        # Update parameters
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        if profile_enabled:
                            prof.step()

            
                        
                        #print('')
                        #print(loss_value.shape)
                        total_train_loss += loss_value.clone().detach()
                        total_diff += loss_n.clone().detach()
                        #print(t1-t0)
                        #print('')
                        #weights[indexbatch] = wv
                        
                        del points, speed, loss_value, loss_n, wv#,indexbatch        
                    

                    total_train_loss /= 5 #dataloader train_loader
                    total_diff /= 5 #dataloader train_loader

                    current_diff = total_diff
                    diff_ratio = current_diff/prev_diff
            
                    if True and (diff_ratio < 3.2 and diff_ratio > 0 or e < 10):#1.5
                        #self.optimizer.param_groups[0]['lr'] = prev_lr 
                        break
                    else:
                        
                        iter+=1
                        with torch.no_grad():
                            random_number = random.randint(0, 4)
                            self.load_state_dict(self.prev_state_queue[random_number], strict=True)
                            self.optimizer.load_state_dict(self.prev_optimizer_queue[random_number])

                        print("RepeatEpoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                            self.epoch, total_diff, self.alpha))
                    
                self.writer.add_scalar('Loss/Train', total_diff.item(), self.epoch)
                self.writer.add_scalar('Epoch Time', time.time()-epoch_start_time, self.epoch)
                # print("Epoch Time: ", time.time()-epoch_start_time)
                    

                #'''
                self.total_train_loss.append(total_train_loss)
                
                beta = 1.0/total_diff
                

                if self.Params['Training']['Use Scheduler (bool)'] == True:
                    self.scheduler.step(total_train_loss)

                if self.epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                    with torch.no_grad():
                        print("Epoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                            self.epoch, total_diff.item(), self.alpha))
                
                if self.epoch % self.Params['Training']['Save Every * Epoch'] == 0:
                    self.plot(self.initial_view, self.epoch, total_diff.item(), self.alpha, cur_data[:, :6].clone().cpu().numpy(), None)
            
            
        return total_diff

    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        return NN.NN.TravelTimes(self, Xp)
    
    def Tau(self, Xp):
        return NN.NN.Tau(self, Xp)

    def Speed(self, Xp):
        return NN.NN.Speed(self, Xp)
    
    def plot_out(self, x):
        x = x.clone().detach().requires_grad_(True)
        features = []
        windows = []
        for i, subnet in enumerate(self.subnets):
            feature, window = subnet.out(x)
            features.append(feature)
            windows.append(window)
        windows = torch.stack(windows, dim=0)
        ws  = torch.sum(windows, dim=0)
        subnet_outputs = []
        for i, raw_encoder in enumerate(features, 1):
            subnet_feature =  raw_encoder * windows[i-1] / ws
            output, _ = self.sym_op(subnet_feature, x)
            subnet_outputs.append(output)
        return subnet_outputs  
    
    def plot_out_valid(self, x, valid_indices):
        x = x.clone().detach().requires_grad_(True)
        features = []
        windows = []
        for i, subnet in enumerate(self.subnets):
            feature, window = subnet.out(x)
            features.append(feature)
            windows.append(window)
        windows = torch.stack(windows, dim=0)
        ws  = torch.sum(windows, dim=0)
        features = torch.stack(features, dim=0)
        valid_features = []
        for indice in valid_indices:
            raw_encoder = features[indice]
            subnet_feature =  raw_encoder * windows[indice]
            valid_features.append(subnet_feature)
        valid_feature = torch.sum(torch.stack(valid_features), dim=0)
        normalized_feature = valid_feature / ws
        output, _ = self.sym_op(normalized_feature, x)
        return output, x
 
    def plot_out_end_valid(self, x, valid_indices):
        #TODO: only filter the end points
        x = x.clone().detach().requires_grad_(True)
        features = []
        windows = []
        for i, subnet in enumerate(self.subnets):
            feature, window = subnet.out(x)
            features.append(feature)
            windows.append(window)
        windows = torch.stack(windows, dim=0)
        ws  = torch.sum(windows, dim=0)
        features = torch.stack(features, dim=0)
        normalized_features = []
        for indice in valid_indices:
            raw_encoder = features[indice]
            subnet_feature =  raw_encoder * windows[i] / ws
            normalized_features.append(subnet_feature)
        normalized_feature = torch.sum(torch.stack(normalized_features), dim=0)
        output, _ = self.sym_op(normalized_feature, x)
        return output, x

    def plot_end_out(self, x):
        x = x.clone().detach().requires_grad_(False)
        size = x.shape[0]
        start_features = []
        end_features = []
        windows = []
        for i, subnet in enumerate(self.subnets):
            feature, window = subnet.module.out(x)
            start_features.append(feature[:size, :])
            end_features.append(feature[size:, :])  
            windows.append(window)
        windows = torch.stack(windows, dim=0)
        ws = torch.sum(windows, dim=0)
        start_windows = windows[:, :size, :]
        end_windows = windows[:, size:, :]
        start_ws = ws[:size, :]
        end_ws = ws[size:, :]

        #? normalize the start features
        for i in range(len(self.subnets)):
            start_features[i] = start_features[i] * start_windows[i] / start_ws
        start_feature = torch.sum(torch.stack(start_features), dim=0)

        #? normalize the end features and output the tau
        outputs = []
        for i in range(len(self.subnets)):
            end_features[i] = end_features[i] * end_windows[i] / end_ws
            feature = torch.cat((start_feature, end_features[i]), dim=0)
            output, _ = self.sym_op(feature, x)
            outputs.append(output)
        return outputs

    def plot(self, src, epoch, total_train_loss, alpha, cur_points=None, camera_matrix=None, traj_list = None):
        # limit = 1
        # xmin = [-0.5, -0.5]
        # xmax = [0.5, 0.5]

        # limit = 5
        # xmin = [-2.5, -2.5]
        # xmax = [2.5, 2.5]

        # limit = 20
        # xmin = [-10, -10]
        # xmax = [10, 10]

        # limit = 6
        # xmin = [-3, -3]
        # xmax = [3, 3]
        offsetxyz = self.config['data']['offset']
        centerxyz = self.config['data']['center']
        xmin = [centerxyz[0]-offsetxyz[0], centerxyz[1]-offsetxyz[1]]
        xmax = [ centerxyz[0]+offsetxyz[0], centerxyz[1]+offsetxyz[1]]
        # xmin = [-offsetxyz[0], -offsetxyz[1]]
        # xmax = [offsetxyz[0], offsetxyz[1]]
        limit = max(offsetxyz[0], offsetxyz[1])*2
        
        # if self.mode == READ_FROM_TURTLEBOT:
        #     xmin = [-1.5, -1.5]
        #     xmax = [1.5, 1.5]
        spacing=limit/80.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))

        # Xsrc = [0]*self.dim
        # Xsrc[0]=-6 #?change
        # Xsrc[1]=3
        # Xsrc[2]=0.3
        # Xsrc = src.detach().clone().cpu().numpy()
        Xsrc = src 
        # Xsrc[2] = 0.2
        # Xsrc = [0.2, -0.2, 0]
        
        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc
        #XP=XP/scale
        XP[:,self.dim+0]  = X.flatten()
        XP[:,self.dim+1]  = Y.flatten() #! this allow to change from y to z
        XP = Variable(Tensor(XP)).to(self.Params['Device'])
        
        tt = self.TravelTimes(XP)
        ss = self.Speed(XP)#*5
        tau = self.Tau(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=1)


        # #! camera triangle
        # if camera_matrix is not None:
        #     orientation_matrix = camera_matrix[:3, :3]  # Assuming the rotation part is the 3x3 upper-left submatrix
        #     position = src

        #     # Calculate the orientation angle from the orientation matrix
        #     yaw = np.arctan2(orientation_matrix[1, 0], orientation_matrix[0, 0])  # You can use this angle to represent the camera look at
        #     yaw += np.pi/2

        #     # Create a small triangle marker for the camera
        #     triangle_size = 0.2  # Adjust the size as needed
        #     triangle_marker = np.array([[0, 0], [triangle_size/2, -triangle_size/2], [triangle_size / 2, triangle_size/2]])
            
        #     # Rotate the triangle marker to match the camera's orientation
        #     rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        #     rotated_triangle_marker = np.dot(triangle_marker, rotation_matrix.T)

        #     # # Add the triangle marker at the camera position
        #     # ax.plot(rotated_triangle_marker[0], rotated_triangle_marker[1], 'r^', markersize=10)

        #     camera_x = position[0] + rotated_triangle_marker[:, 0]
        #     camera_y = position[1] + rotated_triangle_marker[:, 1]
        #     ax.fill(camera_x, camera_y, 'b')

    
        # #! plot trajectory
        # if traj_list is not None:
        #     ax.plot(traj_list[:, 0], traj_list[:, 1], color='pink', marker = 'o', markersize=0.8, linestyle='-')
        #     traj_list_path = self.folder+"/"+str(epoch)+".npy"
        #     np.save(traj_list_path, traj_list)
        #     plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.png",bbox_inches='tight')

        #     #? plot trajectory with step size
        #     ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], color='red', marker='o', markersize=0.8, linestyle='-', linewidth=1)
            

        # ax.contour(X,Y,TT,np.arange(0,5,0.01), cmap='bone', linewidths=0.3)#0.25
        contour_density = self.config["model"]["contour_density"]
        ax.contour(X,Y,TT,np.arange(0,100,contour_density), cmap='bone', linewidths=0.3)#0.25

        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.png",bbox_inches='tight')

        plt.close(fig)
            



        if cur_points is not None:
            # print("plot: cur point shape:",cur_points.shape)
            start_points = cur_points[:,:3]
            end_points = cur_points[:,3:]
            diff = start_points - end_points
            dists = np.linalg.norm(diff, axis=-1)
            # print("dists:", dists.shape)
            plt.hist(dists, bins=100)
            plt.savefig(self.folder+"/plots_dist_"+str(epoch)+".png")
            plt.close()


        #?add four subregion encoder + generator
        if False:
            # Calculate the sum of all weights
            #? pure subnets output
            # weighted_outputs = self.plot_out(XP)
            #? normalized start feature, and show output feature
            weighted_outputs = self.plot_end_out(XP)

            plt.figure(figsize=(12, 10))
            #! loop over the subnets
            cols = self.config["region"]["columns"]
            rows = self.config["region"]["rows"]
            for i, region in enumerate(self.regions, 1):
                weighted_output = weighted_outputs[i - 1].to('cpu').data.reshape(X.shape)
                # change nan to 0
                # weighted_output = torch.where(torch.isnan(weighted_output), torch.zeros_like(weighted_output), weighted_output) 

                plt.subplot(cols, rows, i)
                plt.gca().set_aspect('equal', adjustable='box')
                quad1 = plt.pcolormesh(X, Y, weighted_output.numpy(), vmin=0, vmax=1)
                plt.colorbar(quad1, pad=0.1, label=f'Subnet Output {i}')

            plt.tight_layout()
            plt.savefig(self.folder+"/plots"+str(epoch) + "_subnets.png")
            plt.close()

    def plot_valid(self, src, valid_indices):
        offsetxyz = self.config['data']['offset']
        xmin = [-offsetxyz[0], -offsetxyz[1]]
        xmax = [offsetxyz[0], offsetxyz[1]]
        limit = max(offsetxyz[0], offsetxyz[1])*2
        
        spacing=limit/80.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))

        Xsrc = src 

        
        XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim:] = Xsrc
        #XP=XP/scale
        XP[:,self.dim+0]  = X.flatten()
        XP[:,self.dim+1]  = Y.flatten() #! this allow to change from y to z
        XP = Variable(Tensor(XP)).to(self.Params['Device'])
        
        #! hardcode travel time and speed based on the region
        tau, Xp = self.plot_out_valid(XP, valid_indices)
        tt = tau[:, 0]
        dtau = self.gradient(tau, Xp)
        DT1 = dtau[:, self.dim:]
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
        ss = 1/torch.sqrt(S1)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=1)

        contour_density = self.config["model"]["contour_density"]
        ax.contour(X,Y,TT,np.arange(0,30,contour_density), cmap='bone', linewidths=0.3)#0.25

        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.folder+"/plots_valid"+"_0.png",bbox_inches='tight')

        plt.close(fig)





def main():
    # torch.cuda.memory._record_memory_history(enabled=True)
    modelPath = './Experiments'
    config_path = "configs/cabin.yaml"
    config_path = "configs/maze.yaml"
    config_path = "configs/ruiqi.yaml"
    config_path = "configs/mesh.yaml"
    config_path = "configs/cube_passage.yaml"
    config_path = "configs/b1.yaml"
    config_path = "configs/almena.yaml"
    config_path = "configs/narrow_cube.yaml"
    model    = Model(modelPath, config_path, device='cuda:0')
    model.train()

def eval():
    modelPath = './Experiments'
    config_path = "configs/cube_passage.yaml"
    modelpath = "Experiments/07_09_14_02/Model_Epoch_01500_ValLoss_7.895163e-04.pt"

    model    = Model(modelPath, config_path, device='cuda:0')
    model.load(modelpath)
    model.plot( Tensor([-2.0, -0.0, 0]), 0, 0, 0, None, None, None)
    model.plot_valid( Tensor([-2.0, -0.0, 0]), [ 0, 2, 3,  5, 6, 7, 8])
    
    
    # t = torch.tensor([[-0.3, -0.2, 0, -0.3, -0.2, 0],
    #                   [-0.3, 0.2, 0, -0.3, 0.2, 0],
    #                   [0.3, 0, 0, 0.3, 0, 0],
    #                   [0.3, 0.35, 0, 0.3, 0.35, 0],
    #                   [-0.3, -0.35, 0, -0.3, -0.35, 0],
    #                   [-0.1, -0.3, 0, -0.1, -0.3, 0],]).to('cuda:0')
    # model.plot_out_valid(t, [0, 1, 2, 3, 4, 5, 6, 7])
    # features, _ = model.all_encode_out(t)
    # start_features = features[:6]
    # scales = torch.sqrt(torch.sum (  ( start_features**2 ) , dim =1)).unsqueeze(1) 
    # print(scales)
    # print("done")



if __name__ == "__main__":
    main()
    # eval()


