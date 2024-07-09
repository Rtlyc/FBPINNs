""" 
This script serves as an executing script to transform the original model into a submodel manner. To do this, we need to split the original model into parts: 1. encoder 2. window function 3. symmetric operator 
"""

import torch 
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
    return w_x * w_y

# def scale_region(region, coord):
#     # scale the region to -0.5, 0.5
#     xmin, xmax, ymin, ymax = region
#     kx = (xmax - xmin)

def plot_window_2d_normalized(regions, savepath):
    # xmin, xmax, ymin, ymax = -0.5, 0.5, -0.5, 0.5
    xmin, xmax, ymin, ymax = -2.5, 2.5, -2.5, 2.5
    xmin, xmax, ymin, ymax = -10, 10, -10, 10
    xmin, xmax, ymin, ymax = -3, 3, -3, 3



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
    plt.figure(figsize=(5, 10))
    for i, region in enumerate(regions, 1):
        # Normalize the weights for this region
        normalized_weights = all_weights[i - 1] / total_weights
        # plt.subplot(int(math.sqrt(len(regions))), int(math.sqrt(len(regions))), i)
        plt.subplot(3, 3, i)
        cp = plt.contourf(xx.numpy(), yy.numpy(), normalized_weights.numpy(), levels=50, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(cp, ticks=[0, 1])
        xmin, xmax, ymin, ymax = region
        plt.title(f'Region {i}: [{xmin},{xmax}] x [{ymin},{ymax}]')
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
    
    def encoder_out(self, coords):
        if self.config['model']['normalization']:
            #!: scale the input to -0.5, 0.5
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
        return NN.SubNN.encoder_out(self, coords)
    

    def out(self, coords):
        # ? this should only output encoder features
        size = coords.shape[0]
        encoder_output, _ = self.encoder_out(coords)
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        x_stack = torch.vstack((x0, x1))
        # window = cosine_2d(*self.region, x_stack[:, 0, None], x_stack[:, 1, None])
        window = gaussian_2d(*self.region, x_stack[:, 0, None], x_stack[:, 1, None])
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

        self.init_network()
        self.load_rawdata()





    def init_network(self):
        #! initialize the network, notice that we need to split the original model into submodels
        #? 2D window
        width = 5.0
        height = 5.0
        rows, cols = 3, 3
        row_s = width / rows
        row_s2 = width / rows / 2
        x_start = -width/2 - row_s2
        y_start = width/2 + row_s2
        regions = []
        for ri in range(rows):
            for ci in range(cols):
                x1 = x_start + ci * row_s 
                x2 = x1 + row_s*2 
                y1 = y_start - ri * row_s
                y2 = y1 - row_s*2
                regions.append((x1, x2, y2, y1))
        self.regions = regions

        self.regions = [(-5, 5, -15, -5),
                        (-5, 5, -10, 0),
                        (-5, 5, -5, 5),
                        (-5, 5, -0, 10),
                        (-5, 5, 5, 15),]
        
        # self.regions = [(-10.5, 10.5, -10.5, 10.5)]
        self.regions = [(-5, 5, -5, 5)]

        self.regions = [(-6, 0, 0, 6),
                        (-3, 3, 0, 6),
                        (-0, 6, 0, 6),
                        (-6, 0, -3, 3),
                        (-3, 3, -3, 3),
                        (-0, 6, -3, 3),
                        (-6, 0, -6, 0),
                        (-3, 3, -6, 0),
                        (-0, 6, -6, 0),]
        
        self.regions = self.config["regions"]

        plot_window_2d_normalized(self.regions, self.folder)
        
        self.subnets = torch.nn.ModuleList([SubModel(region, self.config, self.device) for region in self.regions])

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.Params['Training']['Learning Rate'], weight_decay=0.2)

    def load_rawdata(self):
        if False:
            self.explored_data = np.load("data/explored_data.npy")
            #! hardcode room size
            self.explored_data[:, :, :6] *= 5.0
            print("hardcoded room size scale: 5.0")
        else:
            # points = np.load("data/cube_passage_points.npy")
            # speeds = np.load("data/cube_passage_speeds.npy")
            # points = np.load("data/cabin_points.npy")
            # speeds = np.load("data/cabin_speeds.npy")
            points = np.load(self.config["paths"]["pointspath"]).astype(np.float32)
            speeds = np.load(self.config["paths"]["speedspath"]).astype(np.float32)
            self.explored_data = np.concatenate((points, speeds), axis=1)
        print("explored data shape:", self.explored_data.shape)

    def randomize_data(self, frame_data):
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

    def all_encode_out(self, x):
        x = x.clone().detach().requires_grad_(True)
        features, weights = [], []
        for subnet in self.subnets:
            feature, weight = subnet.out(x)
            features.append(feature*weight)
            weights.append(weight)
        features = torch.stack(features)
        weights = torch.stack(weights)
        outputs = torch.sum(features, dim=0)
        ws = torch.sum(weights, dim=0)
        normalized_features = outputs/ws
        return normalized_features, x 
    
    def sym_op(self, normalized_features, Xp):
        return NN.NN.sym_op(self, normalized_features, Xp)



    def out(self, x):
        #! core function to output the normalized features from all the subnets
        normalized_features, Xp = self.all_encode_out(x)
        outputs, Xp = self.sym_op(normalized_features, Xp)
        return outputs, Xp
        

    def gradient(self, y, x, create_graph=True):
        grad_y = torch.ones_like(y)
        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x

    def Loss(self, points, Yobs, beta):
        return NN.NN.Loss(self, points, Yobs, beta)

    def train(self):

        frame_epoch = 50 
        self.alpha = 1.0

        while True:
            if False:
                #? by frame sequence
                frame_data = self.explored_data[self.frame_idx]
                frame_data = torch.tensor(frame_data).to(self.Params['Device'])
            else:
                #? by random 
                explored_data = self.explored_data.reshape(-1, 8)
                rand_idx = torch.randperm(explored_data.shape[0])[:5000*64]
                frame_data = torch.tensor(explored_data[rand_idx]).to(self.Params['Device'])
            total_diff, cur_data = self.train_core(frame_epoch, frame_data)

            self.plot(self.initial_view, self.epoch, total_diff.item(), self.alpha, cur_data[:, :6].clone().cpu().numpy(), None)

            with torch.no_grad():
                self.save(epoch=self.epoch, val_loss=total_diff)

            self.frame_idx += 1
            if self.frame_idx >= len(self.explored_data):
                break

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.subnets.state_dict(),
                    'B_state_dicts': [subnet.B for subnet in self.subnets],
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.folder, str(epoch).zfill(5), val_loss))
        
    def load(self, filepath):
        #B = torch.load(self.Params['ModelPath']+'/B.pt')
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        Bs = checkpoint['B_state_dicts']

        self.subnets = torch.nn.ModuleList([SubModel(region, self.config, self.device) for region in self.regions])
        for i, subnet in enumerate(self.subnets):
            subnet.B = Bs[i]

        self.subnets.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.subnets.to(torch.device(self.device))
        self.subnets.float()
        self.subnets.eval()


    def train_core(self, epoch, frame_data):
        beta = 1.0
        prev_diff = 1.0
        current_diff = 10.0
        step = -1000.0/4000.0
        if self.all_framedata is None:
            self.all_framedata = frame_data.unsqueeze(0)
        else:
            self.all_framedata = torch.cat((self.all_framedata, frame_data.unsqueeze(0)), dim=0)
        print(self.all_framedata.shape)


        #! mix data so that the start and end points are from different frames
        # if is_one_frame:
        cur_data = self.randomize_data(frame_data)
        dataloader = FastTensorDataLoader(cur_data, batch_size=int(self.Params['Training']['Batch Size']), shuffle=True)

        frame_epoch = epoch
        # start_time = time.time()
        for e in range(frame_epoch):
            self.epoch += 1
            total_train_loss = 0
            total_diff=0

            gamma=0.001#max((4000.0-epoch)/4000.0/20,0.001)
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


                    loss_value, loss_n, wv = self.Loss(
                    points, speed, beta)
                    loss_value.backward()

                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
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
                
                
            #'''
            self.total_train_loss.append(total_train_loss)
            
            beta = 1.0/total_diff
            

            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_train_loss)

            if self.epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    print("Epoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        self.epoch, total_diff.item(), self.alpha))
        return total_diff, cur_data

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

    def plot_end_out(self, x):
        x = x.clone().detach().requires_grad_(False)
        size = x.shape[0]
        start_features = []
        end_features = []
        windows = []
        for i, subnet in enumerate(self.subnets):
            feature, window = subnet.out(x)
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
        xmin = [-offsetxyz[0], -offsetxyz[1]]
        xmax = [offsetxyz[0], offsetxyz[1]]
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
        ax.contour(X,Y,TT,np.arange(0,30,0.02), cmap='bone', linewidths=0.3)#0.25

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
        if True:
            # Calculate the sum of all weights
            #? pure subnets output
            # weighted_outputs = self.plot_out(XP)
            #? normalized start feature, and show output feature
            weighted_outputs = self.plot_end_out(XP)

            plt.figure(figsize=(12, 10))
            #! loop over the subnets
            for i, region in enumerate(self.regions, 1):
                weighted_output = weighted_outputs[i - 1].to('cpu').data.reshape(X.shape)
                # change nan to 0
                # weighted_output = torch.where(torch.isnan(weighted_output), torch.zeros_like(weighted_output), weighted_output) 

                plt.subplot(3, 3, i)
                plt.gca().set_aspect('equal', adjustable='box')
                quad1 = plt.pcolormesh(X, Y, weighted_output.numpy(), vmin=0, vmax=1)
                plt.colorbar(quad1, pad=0.1, label=f'Subnet Output {i}')

            plt.tight_layout()
            plt.savefig(self.folder+"/plots"+str(epoch) + "_subnets.png")
            plt.close()





def main():
    # torch.cuda.memory._record_memory_history(enabled=True)
    modelPath = './Experiments'
    scale_factor = 10
    config_path = "configs/cabin.yaml"
    config_path = "configs/ruiqi.yaml"
    model    = Model(modelPath, config_path, device='cuda:0')
    model.train()

def eval():
    modelPath = './Experiments'
    scale_factor = 10
    model    = Model(modelPath, 3, scale_factor, device='cuda:0')
    modelpath = "Experiments/06_27_10_55/Model_Epoch_02200_ValLoss_2.624636e-02.pt"
    model.load(modelpath)
    model.plot( Tensor([-0.3, -0.2, 0]), 0, 0, 0, None, None, None)
    t = torch.tensor([[-0.3, -0.2, 0, -0.3, -0.2, 0],
                      [-0.3, 0.2, 0, -0.3, 0.2, 0],
                      [0.3, 0, 0, 0.3, 0, 0],
                      [0.3, 0.35, 0, 0.3, 0.35, 0],
                      [-0.3, -0.35, 0, -0.3, -0.35, 0],
                      [-0.1, -0.3, 0, -0.1, -0.3, 0],]).to('cuda:0')
    features, _ = model.all_encode_out(t)
    start_features = features[:6]
    scales = torch.sqrt(torch.sum (  ( start_features**2 ) , dim =1)).unsqueeze(1) 
    print(scales)
    print("done")



if __name__ == "__main__":
    main()
    # eval()


