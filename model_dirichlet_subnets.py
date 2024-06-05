import numpy as np
import math
import random
import time
import os
from datetime import datetime, timedelta

import torch
from torch.nn import Linear
from torch import Tensor
from torch.autograd import Variable, grad
from torchvision import transforms


import matplotlib
import matplotlib.pylab as plt

import pickle5 as pickle 

from timeit import default_timer as timer

torch.backends.cudnn.benchmark = True

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

def sigmoid(input):
 
    return torch.sigmoid(10*input)

class Sigmoid(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid(input) 

class DSigmoid(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 10*sigmoid(input)*(1-sigmoid(input)) 

def sigmoid_out(input):
 
    return torch.sigmoid(0.1*input)

class Sigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return sigmoid_out(input) 

class DSigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 0.1*sigmoid_out(input)*(1-sigmoid_out(input)) 

class DDSigmoid_out(torch.nn.Module):
    def __init__(self):
        
        super().__init__() 

    def forward(self, input):
       
        return 0.01*sigmoid_out(input)*(1-sigmoid_out(input))*(1-2*sigmoid_out(input))


class NN(torch.nn.Module):
    
    def __init__(self, device, dim ,B):#10
        super(NN, self).__init__()
        self.dim = dim

        h_size = 128 #512,256
        #input_size = 128
        #self.T=2

        self.B = B.T.to(device)
        print(B.shape)
        input_size = B.shape[0]
        #decoder
        self.scale = 10

        self.act = torch.nn.Softplus(beta=self.scale)#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU
        #self.ddact = torch.nn.Sigmoid()-torch.nn.Sigmoid()*torch.nn.Sigmoid()
        self.actout = Sigmoid_out()#ELU,CELU

        #self.env_act = torch.nn.Sigmoid()#ELU

        self.nl1=3
        self.nl2=2

        self.encoder = torch.nn.ModuleList()
        self.encoder1 = torch.nn.ModuleList()
        #self.encoder.append(Linear(self.dim,h_size))
        
        self.encoder.append(Linear(2*input_size,h_size))
        self.encoder1.append(Linear(2*input_size,h_size))
        
        for i in range(self.nl1-1):
            self.encoder.append(Linear(h_size, h_size)) 
            self.encoder1.append(Linear(h_size, h_size)) 
        
        self.encoder.append(Linear(h_size, h_size)) 

        self.generator = torch.nn.ModuleList()
        self.generator1 = torch.nn.ModuleList()
        for i in range(self.nl2):
            self.generator.append(Linear(h_size, h_size)) 
            self.generator1.append(Linear(2*h_size, 2*h_size)) 
        
        self.generator.append(Linear(h_size,h_size))
        self.generator.append(Linear(h_size,1))
    #'''
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2
            #stdv = np.sqrt(6 / 64.) / self.T
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)
    #'''
    def input_mapping(self, x):
        w = 2.*np.pi*self.B
        x_proj = x @ w
        #x_proj = (2.*np.pi*x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)    #  2*len(B)

    def out(self, coords):
        
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))

        #x = x.unsqueeze(1)
        
        x = self.input_mapping(x)
        x  = torch.sin(self.encoder[0](x))
        for ii in range(1,self.nl1):
            #i0 = x
            #x_tmp = x
            x  = torch.sin(self.encoder[ii](x))
            #x  = self.act(self.encoder1[ii](x) + x_tmp) 
        
        x = self.encoder[-1](x)

        x0 = torch.sin(x[:size,...])#torch.sin
        x1 = torch.sin(x[size:,...])#torch.sin

        #xx = torch.cat((x0, x1), dim=1)
        
        #x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        #x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale

        #x = torch.cat((x_0, x_1),1)

        x = (x0*x1)
        #(x0[:,:128]-x1[:,:128])**2
        #*x0[:,:128]*x1[:,:128]#
        #xx = torch.cat((x0-x1, x1-x0), dim=1)
        
        #x = torch.logsumexp(xx, 1)
        #'''
        for ii in range(self.nl2):
            #x_tmp = x
            x = torch.sin(self.generator[ii](x)) 
            #w = self.generator[ii].weight
            #x = torch.sin(x@w.T)
            #x = self.act(self.generator1[ii](x) + x_tmp) 
        
        y = self.generator[-2](x)
        x = torch.sin(y)

        #w = self.generator[-2].weight
        #x = torch.sin(x@w.T)
        #'''
        #y = torch.sum(x,dim=1).unsqueeze(1)#
        
        y = self.generator[-1](x)
        x = self.actout(y)#/(math.e)

        #w = self.generator[-1].weight
        #x = (1-torch.exp(-10*(x@w.T)**2))/(math.e)
        #x = self.actout(y*torch.tanh(0.1*y))/(math.e)
        #print(x.shape)
        #output = output.squeeze(2)
        
        return x, coords

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

        output, coords = self.out(coords)
        return output, coords


class Model():
    def __init__(self, ModelPath, dim, scale_factor, device='cpu'):

        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.dim = dim
        self.scale_factor = scale_factor
        current_time = datetime.utcnow()-timedelta(hours=5)
        self.folder = self.Params['ModelPath']+"/"+current_time.strftime("%m_%d_%H_%M")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            # os.chmod(self.folder, 0o777)

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

        # # Parameters for the occupancy grid
        dim_cells = 100
        limit = 1.05
        spacing = limit/dim_cells
        # self.occ_map = gridmap.OccupancyGridMap(np.zeros((dim_cells,dim_cells)), cell_size=spacing, occupancy_threshold=0.8, offset=limit/2)

        # self.mode = mode
        self.frame_buffer_size = 20
        self.camera_steps = 5000//50
        self.minimum = 0.01 #0.02
        self.maximum = 0.1  #0.1
        self.all_framedata = None
        self.all_surf_pc = []
        self.free_pc = []

        self.init_network()
        

    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y)                                                                 

        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        
        return grad_x                                                                                                    

    def Loss(self, points, Yobs, beta, gamma):
        
      
        start=time.time()
        tau, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)
        end=time.time()
        
        #print(end-start)

        start=time.time()
        
        
        #tau, dtau, Xp = self.network.out_grad(points)
        
        end=time.time()
        #print(end-start)
        #print(dtau)

        #print(end-start)
        #print('')
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)#torch.norm(D, p=2, dim =1)**2
        
        
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]

        T3    = tau[:,0]**2

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2
        LogTau3 = LogTau**3
        LogTau4 = LogTau**4

        #print(tau.shape)

        T01    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT0, DT0)
        T02    = -4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT0, D)
        #print(T02.shape)
        T11    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT1, DT1)
        T12    = 4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        lap0 = 0.5*torch.einsum('ij,ij->i', DT0, DT0)#ltau[:,:self.dim].sum(-1) 
        lap1 = 0.5*torch.einsum('ij,ij->i', DT1, DT1)#ltau[:,self.dim:].sum(-1) 
        
        
        
        S0 = (T01+T02+LogTau4)
        S1 = (T11+T12+LogTau4)
       
        #0.001
        Ypred0 = torch.sqrt(S0)#torch.sqrt
        Ypred1 = torch.sqrt(S1)#torch.sqrt


        Ypred0_visco = Ypred0
        Ypred1_visco = Ypred1

        sq_Ypred0 = (Ypred0_visco)#+gamma*lap0
        sq_Ypred1 = (Ypred1_visco)#+gamma*lap1


        sq_Yobs0 = (Yobs[:,0])#**2
        sq_Yobs1 = (Yobs[:,1])#**2

        #loss0 = (sq_Yobs0/sq_Ypred0+sq_Ypred0/sq_Yobs0)#**2#+gamma*lap0
        #loss1 = (sq_Yobs1/sq_Ypred1+sq_Ypred1/sq_Yobs1)#**2#+gamma*lap1
        l0 = ((sq_Yobs0*(sq_Ypred0)))
        l1 = ((sq_Yobs1*(sq_Ypred1)))
        
        l0_2 = (torch.sqrt(l0))#**(1/4)
        l1_2 = (torch.sqrt(l1))#**(1/4)
        l0_4 = torch.sqrt(l0_2)#**(1/4)
        l1_4 = torch.sqrt(l1_2)#**(1/4)

        relu_loss0 = 10*torch.nn.functional.relu(1/sq_Ypred0-1)
        relu_loss1 = 10*torch.nn.functional.relu(1/sq_Ypred1-1)
        loss0 = (l0_2-1)**2#+relu_loss0#**2#+gamma*lap0#**2
        loss1 = (l1_2-1)**2#+relu_loss1#**2#+gamma*lap1#**2
        #print(torch.sum(relu_loss0+relu_loss1))

        #diff = loss0 + loss1-4
        #loss_n = torch.sum((loss0 + loss1-4))/Yobs.shape[0]
        #weight0 = torch.exp(1-Yobs[:,0])/2
        #weight1 = torch.exp(1-Yobs[:,1])/2
        diff = loss0 + loss1 
        loss_n = torch.sum((loss0 + loss1+0.01*lap0+0.01*lap1))/Yobs.shape[0]

        #+0.01*tau[:,0]+0.01*lap0+0.01*lap1
        loss = beta*loss_n #+ 1e-4*(reg_tau)

        return loss, loss_n, diff


    def init_network(self):
        #! Initialising the network
        #self._init_network()
        self.B = 0.5*torch.normal(0,1,size=(128,self.dim)) #0.5
        #torch.save(B, self.Params['ModelPath']+'/B.pt')

        self.network = NN(self.Params['Device'],self.dim, self.B)
        self.network.apply(self.network.init_weights)
        #self.network.float()
        self.network.to(self.Params['Device'])

        #! Defining the optimization scheme
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate']
            ,weight_decay=0.2)
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,2000], gamma=0.5)

    def load_rawdata(self):
        #! load data
        # initial_view = Tensor([-0.3, -0.2, 0])
        # initial_view = Tensor([0.1, 0.2, 0])
        # initial_view = Tensor([-0.2, -0.1, 0])
        # initial_view = Tensor([-0.4, 0.1, 0])
        # initial_view = Tensor([-0, -0., 0])
        # initial_view = Tensor([0.25, -0.2, 0])
        # initial_view = Tensor([0.05, 0., 0]) #branford
        # initial_view = Tensor([0.25, -0.25, 0]) #brevort
        # initial_view = Tensor([-0., -0.25, 0]) #bolton

        initial_view = Tensor([-0.3, -0.2, 0])
        self.initial_view = initial_view
        
        self.cur_view = initial_view

    def randomize_data(self, frame_data):
        prev_frame_idx = torch.randperm(self.frame_idx).tolist()[:self.frame_buffer_size-1]
        prev_framedata = self.all_framedata[prev_frame_idx]
        chosen_framedata = torch.cat((prev_framedata, frame_data.unsqueeze(0)), dim=0).view(-1, 8)

        chosen_points = chosen_framedata[:,:6]
        chosen_speeds = chosen_framedata[:,6:]
        
        points = frame_data[:,:6]
        speeds = frame_data[:,6:]
        if False: # ray sampling method
            local_start_index = torch.randperm(points.shape[0])[:int(points.shape[0]*1.0)]
            local_end_index = torch.randperm(points.shape[0])[:int(points.shape[0]*1.0)]
            local_points_combination = torch.cat((points[local_start_index,:3], points[local_end_index,3:]), dim=1)
            local_speeds_combination = torch.cat((speeds[local_start_index,:1], speeds[local_end_index,1:]), dim=1)
        else: # random sampling method
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

    def train(self):
        self.load_rawdata()
        explored_data = np.load("/home/exx/Documents/p-nt-share/Experiments/05_29_13_53/explored_data.npy")
        while True:
            if True:
                #? by frame sequence
                frame_data = explored_data[self.frame_idx]
                frame_data = torch.tensor(frame_data).to(self.Params['Device'])
            else:
                #? by random 
                explored_data = explored_data.reshape(-1, 8)
                rand_idx = torch.randperm(explored_data.shape[0])[:5000]
                frame_data = torch.tensor(explored_data[rand_idx]).to(self.Params['Device'])

            frame_epoch = 50

            ####################! Core Training #############################
            self.alpha = 1.025
            total_diff, cur_data = self.train_core(frame_epoch, frame_data)

            self.plot(self.initial_view, self.epoch, total_diff.item(),self.alpha, cur_data[:,:6].clone().cpu().numpy(), None)

            
            # self.fourplot(epoch, FRAMES[:frame_idx+1], total_diff.item(), alpha)
            with torch.no_grad():
                self.save(epoch=self.epoch, val_loss=total_diff)

            #? ******************SAVINGS end*******************
            
            self.frame_idx += 1
            if self.frame_idx >= self.camera_steps:
                break
            
    def train_core(self, epoch, frame_data):
        beta = 1.0
        prev_diff = 1.0
        current_diff = 10.0
        step = -1000.0/4000.0
        tt =time.time()

        if self.all_framedata is None:
            self.all_framedata = frame_data.unsqueeze(0)
        else:
            self.all_framedata = torch.cat((self.all_framedata, frame_data.unsqueeze(0)), dim=0)
        print(self.all_framedata.shape)
            # np.save(f"{self.folder}/explored_data.npy", self.all_framedata.clone().cpu().numpy())


        #! mix data so that the start and end points are from different frames
        # if is_one_frame:
        cur_data = self.randomize_data(frame_data)
            # print("cur_data:", cur_data.shape)
        # else:
        #     rand_idx = torch.randperm(frame_data.shape[0])
        #     chosen_data = frame_data[rand_idx][:100000]
        #     chosen_points = chosen_data[:,:6]
        #     chosen_speeds = chosen_data[:,6:]
            
        #     chosen_percent = 1.0
        #     # print("chosen points shape:", chosen_points.shape)
        #     chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        #     chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        #     # print("chosen points indice shape:", chosen_start_index.shape)
        #     chosen_start_points = chosen_points[chosen_start_index][:, :3]
        #     chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
        #     chosen_end_points = chosen_points[chosen_end_index][:, 3:6]
        #     chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]

        #     cur_data = torch.cat((chosen_start_points, chosen_end_points, chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)
                
        # if self.mode == READ_FROM_COOKED_DATA: #? use all framedata
        #     dataloader = FastTensorDataLoader(self.all_framedata, batch_size=int(self.Params['Training']['Batch Size']), 
        #     shuffle=True)
        # else: #? use current framedata continual learning
        dataloader = FastTensorDataLoader(cur_data, batch_size=int(self.Params['Training']['Batch Size']), shuffle=True)


        # frame_epoch = 10000 if isOffline else 50
        frame_epoch = epoch
        # start_time = time.time()
        for e in range(frame_epoch):
            self.epoch += 1
            total_train_loss = 0
            total_diff=0

            # alpha = min(max(0.85,0.85+0.5*step),0.9)
            # alpha = 1.10#1.05
            

            # step+=1.0/4000/((int)(self.epoch/4000)+1.) * 2.0
            # gamma=0.001#max((4000.0-epoch)/4000.0/20,0.001)

            # alpha = 1.025#min(max(0.9,0.9+0.1*step),0.95)
            step+=1.0/4000/((int)(epoch/4000)+1.)
            gamma=0.001#max((4000.0-epoch)/4000.0/20,0.001)
            # mu = 10

            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
            self.prev_state_queue.append(current_state)
            self.prev_optimizer_queue.append(current_optimizer)
            if(len(self.prev_state_queue)>5):
                self.prev_state_queue.pop(0)
                self.prev_optimizer_queue.pop(0)
            
            #self.optimizer.param_groups[0]['lr']  = np.clip(1e-3*(1-(epoch-8000)/1000.), a_min=5e-4, a_max=1e-3) 
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
                    t0 = time.time()
    
                    data=data[0].to(self.Params['Device'])
                    #data, indexbatch = data
                    points = data[:,:2*self.dim]#.float()#.cuda()
                    speed = data[:,2*self.dim:]#.float()#.cuda()
                    
                    speed = speed*speed*(2-speed)*(2-speed)
                    speed = self.alpha*speed+1-self.alpha


                    loss_value, loss_n, wv = self.Loss(
                    points, speed, beta, gamma)
                    loss_value.backward()

                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    #print('')
                    #print(loss_value.shape)
                    total_train_loss += loss_value.clone().detach()
                    total_diff += loss_n.clone().detach()
                    t1 = time.time()
                    #print(t1-t0)
                    #print('')
                    #weights[indexbatch] = wv
                    
                    del points, speed, loss_value, loss_n, wv#,indexbatch
                

                total_train_loss /= 5 #dataloader train_loader
                total_diff /= 5 #dataloader train_loader

                current_diff = total_diff
                diff_ratio = current_diff/prev_diff
        
                if True and (diff_ratio < 1.2 and diff_ratio > 0 or e < 10):#1.5
                    #self.optimizer.param_groups[0]['lr'] = prev_lr 
                    break
                else:
                    
                    iter+=1
                    with torch.no_grad():
                        random_number = random.randint(0, 4)
                        self.network.load_state_dict(self.prev_state_queue[random_number], strict=True)
                        self.optimizer.load_state_dict(self.prev_optimizer_queue[random_number])

                    print("RepeatEpoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        self.epoch, total_diff, self.alpha))
                
                
            #'''
            self.total_train_loss.append(total_train_loss)
            
            beta = 1.0/total_diff
            
            t_1=time.time()
            #print(t_1-t_0)

            #del train_loader_wei, train_sampler_wei

            if self.Params['Training']['Use Scheduler (bool)'] == True:
                self.scheduler.step(total_train_loss)

            t_tmp = tt
            tt=time.time()
            #print(tt-t_tmp)
            #print('')
            if self.epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    #print("Epoch = {} -- Training loss = {:.4e} -- Validation loss = {:.4e}".format(
                    #    epoch, total_train_loss, total_val_loss))
                    print("Epoch = {} -- Loss = {:.4e} -- Alpha = {:.4e}".format(
                        self.epoch, total_diff.item(), self.alpha))
        # end_time = time.time()
        # if is_one_frame:
            # duration = end_time-start_time
            # print(f"running in {duration} secs")
            # self.timer.append(duration)
            # np.save("gib6_time.npy",np.array(self.timer))
        return total_diff, cur_data

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'B_state_dict':self.B,
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_val_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.folder, str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        #B = torch.load(self.Params['ModelPath']+'/B.pt')
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))
        self.B = checkpoint['B_state_dict']

        self.network = NN(self.Params['Device'],self.dim,self.B)

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

    def load_traj(self, filepath):
        self.trajectory = np.load(filepath)
     
    def Gradient(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
       
        #Xp.requires_grad_()
        
        #tau, dtau, coords = self.network.out_grad(Xp)

        tau, Xp = self.network.out2(Xp)
        dtau = self.gradient(tau, Xp)
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D))

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2

        A = 2*LogTau*T0/tau[:,0]
        B = LogTau2/T0
        #print(A.shape)

        Ypred0 = -A*dtau[:,:self.dim]+B*D
        #print(Ypred0.shape)
        Spred0 = torch.norm(Ypred0)
        Ypred0 = 1/Spred0**2*Ypred0

        Ypred1 = -A*dtau[:,self.dim:]-B*D
        Spred1 = torch.norm(Ypred1)

        Ypred1 = 1/Spred1**2*Ypred1
        
        return torch.cat((Ypred0, Ypred1),dim=1)


    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.network.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT
    
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.network.out(Xp)

        return tau

    def Speed(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))

        tau, Xp = self.network.out(Xp)
        dtau = self.gradient(tau, Xp)
        #Xp.requires_grad_()
        #tau, dtau, coords = self.network.out_grad(Xp)
        
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.einsum('ij,ij->i', D, D)

        DT1 = dtau[:,self.dim:]

        T3    = tau[:,0]**2

        LogTau = torch.log(tau[:,0])
        LogTau2 = LogTau**2
        LogTau3 = LogTau**3
        LogTau4 = LogTau**4

        #print(tau.shape)

        #print(T02.shape)
        T1    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT1, DT1)
        T2    = 4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        
        S = (T1+T2+LogTau4)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3
        return Ypred
    

    def plot(self, src, epoch, total_train_loss, alpha, cur_points=None, camera_matrix=None, traj_list = None):
        limit = 1
        xmin = [-0.5, -0.5]
        xmax = [0.5, 0.5]
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


        #! camera triangle
        if camera_matrix is not None:
            orientation_matrix = camera_matrix[:3, :3]  # Assuming the rotation part is the 3x3 upper-left submatrix
            position = src

            # Calculate the orientation angle from the orientation matrix
            yaw = np.arctan2(orientation_matrix[1, 0], orientation_matrix[0, 0])  # You can use this angle to represent the camera look at
            yaw += np.pi/2

            # Create a small triangle marker for the camera
            triangle_size = 0.2  # Adjust the size as needed
            triangle_marker = np.array([[0, 0], [triangle_size/2, -triangle_size/2], [triangle_size / 2, triangle_size/2]])
            
            # Rotate the triangle marker to match the camera's orientation
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rotated_triangle_marker = np.dot(triangle_marker, rotation_matrix.T)

            # # Add the triangle marker at the camera position
            # ax.plot(rotated_triangle_marker[0], rotated_triangle_marker[1], 'r^', markersize=10)

            camera_x = position[0] + rotated_triangle_marker[:, 0]
            camera_y = position[1] + rotated_triangle_marker[:, 1]
            ax.fill(camera_x, camera_y, 'b')

    
        #! plot trajectory
        if traj_list is not None:
            ax.plot(traj_list[:, 0], traj_list[:, 1], color='pink', marker = 'o', markersize=0.8, linestyle='-')
            traj_list_path = self.folder+"/"+str(epoch)+".npy"
            np.save(traj_list_path, traj_list)
            plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.png",bbox_inches='tight')

            #? plot trajectory with step size
            ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], color='red', marker='o', markersize=0.8, linestyle='-', linewidth=1)
            

        ax.contour(X,Y,TT,np.arange(0,5,0.02), cmap='bone', linewidths=0.3)#0.25
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

    def predict_trajectory2(self, Xsrc, Xtar, step_size=0.03, tol=0.03):
        Xsrc = Tensor(Xsrc)
        Xtar = Tensor(Xtar)
        XP_traj= torch.cat((Xsrc,Xtar))
        dis= torch.norm(XP_traj[:self.dim]-XP_traj[self.dim:])
        point_start = []
        point_goal = []
        iter = 0
        while dis > tol:
            gradient = self.Gradient(XP_traj[None,:].clone())
            XP_traj = XP_traj + step_size * gradient[0].cpu()
            XP_traj[2] = Xsrc[2]
            XP_traj[5] = Xsrc[2]
            dis = torch.norm(XP_traj[3:6]-XP_traj[0:3])
            point_start.append(XP_traj[0:3])
            point_goal.append(XP_traj[3:6])
            iter = iter + 1
            if iter > 200:
                break 
        
        point_goal.reverse()
        point = point_start + point_goal
        if point_start:
            traj = torch.cat(point).view(-1, 3)
            traj = torch.cat((Xsrc[None,], traj, Xtar[None,]), dim=0)
        else:
            traj = torch.cat((Xsrc[None,], Xtar[None,]), dim=0)
        # traj = np.concatenate([Xsrc[None,], traj, Xtar[None,]], axis=0)

        return traj.detach().numpy()
    

    def predict_trajectory(self, Xsrc, Xtar, step_size=0.03, tol=0.02):
        Xsrc = Tensor(Xsrc).cuda()
        Xtar = Tensor(Xtar).cuda()
        XP_traj= torch.cat((Xsrc,Xtar))
        dis= torch.norm(XP_traj[:self.dim]-XP_traj[self.dim:])
        point_start = []
        point_goal = []
        iter = 0
        while dis > tol:
            gradient = self.Gradient(XP_traj[None,:].clone())
            XP_traj = XP_traj + step_size * gradient[0]#.cpu()
            dis = torch.norm(XP_traj[3:6]-XP_traj[0:3])
            point_start.append(XP_traj[0:3])
            point_goal.append(XP_traj[3:6])
            iter = iter + 1
            if iter > 100:
                break 
        
        point_goal.reverse()
        point = point_start + point_goal
        if point_start:
            traj = torch.cat(point).view(-1, 3)
            traj = torch.cat((Xsrc[None,], traj, Xtar[None,]), dim=0)
        else:
            traj = torch.cat((Xsrc[None,], Xtar[None,]), dim=0)
        # traj = np.concatenate([Xsrc[None,], traj, Xtar[None,]], axis=0)

        return traj.detach().cpu().numpy()
    


if __name__ == "__main__":

    modelPath = './Experiments'
    scale_factor = 10
    model    = Model(modelPath, 3, scale_factor, device='cuda:0')

    model.train()