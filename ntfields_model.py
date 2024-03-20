import matplotlib
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
# from torchvision import transforms

import matplotlib
import matplotlib.pylab as plt

import pickle5 as pickle 

from nn import NN
from fb_data import generate_sphere_data, FastTensorDataLoader


torch.backends.cudnn.benchmark = True

EXPLORATION = 1 
READ_FROM_DEPTH = 2 
READ_FROM_TURTLEBOT = 3
READ_FROM_COOKED_DATA = 4
TURTLEBOT_EXPLORATION = 5
TOY_FB = 7


class Model():
    def __init__(self, ModelPath, dim, scale_factor, mode, renderer=None, device='cpu'):

        # ======================= JSON Template =======================
        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        # self.Params['DataPath'] = DataPath
        self.renderer = renderer
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
        self.Params['Training']['Save Every * Epoch'] = 20
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

        self.mode = mode
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


        Ypred0_visco = Ypred0#
        Ypred1_visco = Ypred1#+gamma*lap1

        sq_Ypred0 = (Ypred0_visco)#+gamma*lap0
        sq_Ypred1 = (Ypred1_visco)#+gamma*lap1


        sq_Yobs0 = (Yobs[:,0])#**2
        sq_Yobs1 = (Yobs[:,1])#**2

        # loss0 = (sq_Yobs0/sq_Ypred0+sq_Ypred0/sq_Yobs0)#**2#+gamma*lap0
        # loss1 = (sq_Yobs1/sq_Ypred1+sq_Ypred1/sq_Yobs1)#**2#+gamma*lap1

        # diff = loss0 + loss1-4
        # loss_n = torch.sum((loss0 + loss1-4))/Yobs.shape[0]
        l0 = sq_Yobs0*(sq_Ypred0)
        l1 = sq_Yobs1*(sq_Ypred1)
        # l0 = torch.sqrt(torch.sqrt(l0))
        # l1 = torch.sqrt(torch.sqrt(l1))

        l0_2 = torch.sqrt(l0)
        l1_2 = torch.sqrt(l1)
        loss0 = (l0_2-1)**2#+gamma*lap0#**2
        loss1 = (l1_2-1)**2

        # loss0 = abs(l0-1)#+gamma*lap0#**2
        # loss1 = abs(l1-1)

        # loss0 = (l0+1/l0-2)#+gamma*lap0#**2
        # loss1 = (l1+1/l1-2)
        diff = loss0 + loss1
        loss_n = torch.sum((loss0 + loss1))/Yobs.shape[0]

        
        loss = beta*loss_n #+ 1e-4*(reg_tau)

        return loss, loss_n, diff

    def init_network(self):
        #! Initialising the network
        #self._init_network()
        self.B = 0.5 * torch.normal(0,1,size=(128,self.dim)) #0.5
        #torch.save(B, self.Params['ModelPath']+'/B.pt')

        self.network = NN(self.Params['Device'],self.dim, self.B)
        self.network.apply(self.network.init_weights)
        #self.network.float()
        self.network.to(self.Params['Device'])

        #! Defining the optimization scheme
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate']
            ,weight_decay=0.1)
        if self.Params['Training']['Use Scheduler (bool)'] == True:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
            #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,2000], gamma=0.5)

    def load_rawdata(self):
        #! load data
        initial_view = Tensor([-0.3, -0.2, 0])
        # initial_view = Tensor([0.1, 0.2, 0])
        # initial_view = Tensor([-0.2, -0.1, 0])
        # initial_view = Tensor([-0.4, 0.1, 0])
        # initial_view = Tensor([-0, -0., 0])
        # initial_view = Tensor([0.25, -0.2, 0])
        # initial_view = Tensor([0.05, 0., 0]) #branford
        # initial_view = Tensor([0.25, -0.25, 0]) #brevort
        # initial_view = Tensor([-0., -0.25, 0]) #bolton

        if self.mode == READ_FROM_DEPTH:
            experiment_path = "Experiments/01_31_16_20"
            root_dir = os.path.join(experiment_path, "train_data")
            traj_file = os.path.join(root_dir, "traj.txt")
            self.dataset = isdf_sample.IGibsonDataset(root_dir=root_dir, traj_file=traj_file, config='configs/igibson_info.json', col_ext='.png')
            initial_view = Tensor(self.dataset.Ts[0][:3, 3]/self.scale_factor)

        elif self.mode == READ_FROM_TURTLEBOT:
            self.dataset = isdf_sample.TurtleBotDataset(root_dir='turtlebot_data_02_07', config='configs/turtlebot4_info.json', col_ext='.jpg')
            initial_view = Tensor(self.dataset.Ts[0][:3, 3]/self.scale_factor)
        
        elif self.mode == READ_FROM_COOKED_DATA: # read from file
            if True:
                all_framedata = np.load("all_framedata.npy").reshape(-1, 8)
                all_points = all_framedata[:, :6]
                all_speeds = all_framedata[:, 6:]
            else:
                all_points = np.load("mypoints.npy")
                all_speeds = np.load("myspeeds.npy")

            all_points    = Variable(Tensor(all_points))
            all_speeds  = Variable(Tensor(all_speeds))
            print("all_points:", all_points.shape)
            print("all_speeds:", all_speeds.shape)
            self.all_framedata = torch.cat((all_points.to(self.Params['Device']), all_speeds.to(self.Params['Device'])), dim=1)

        elif self.mode == TOY_FB:
            initial_view = Tensor([-0.4, -0.4])

        self.cur_view = initial_view

    def randomize_data(self, frame_data):
        prev_frame_idx = torch.randperm(self.frame_idx).tolist()[:self.frame_buffer_size-1]
        prev_framedata = self.all_framedata[prev_frame_idx]
        chosen_framedata = torch.cat((prev_framedata, frame_data.unsqueeze(0)), dim=0).view(-1, 2*self.dim+2)

        chosen_points = chosen_framedata[:,:2*self.dim]
        chosen_speeds = chosen_framedata[:,2*self.dim:]
        
        points = frame_data[:,:2*self.dim]
        speeds = frame_data[:,2*self.dim:]
        if False: # ray sampling method
            local_start_index = torch.randperm(points.shape[0])[:int(points.shape[0]*1.0)]
            local_end_index = torch.randperm(points.shape[0])[:int(points.shape[0]*1.0)]
            local_points_combination = torch.cat((points[local_start_index,:3], points[local_end_index,3:]), dim=1)
            local_speeds_combination = torch.cat((speeds[local_start_index,:1], speeds[local_end_index,1:]), dim=1)
        else: # random sampling method
            local_points_combination = torch.cat((points[:,:self.dim], points[:,self.dim:]), dim=1)
            local_speeds_combination = torch.cat((speeds[:,:1], speeds[:,1:]), dim=1)

        chosen_percent = 1.0
        # print("chosen points shape:", chosen_points.shape)
        chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
        # print("chosen points indice shape:", chosen_start_index.shape)
        chosen_start_points = chosen_points[chosen_start_index][:, :self.dim]
        chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
        chosen_end_points = chosen_points[chosen_end_index][:, self.dim:2*self.dim]
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
        self.alpha = 1
        while True:
            print("Current Viewpoint:", self.cur_view)

            if self.mode == READ_FROM_COOKED_DATA:
                total_diff, cur_data = self.train_core(100, self.all_framedata, False)
            
            else:
                #! generate data start
                if self.mode == EXPLORATION:
                    heights = [0]
                    curpoints = [] 
                    curspeeds = []
                    curbounds = []
                    for height in heights:
                        curviewpoint = self.cur_view.clone()
                        curviewpoint[2] = height
                        points, speeds, bounds = igib_runner.sample_points_and_speeds_from_pos_new(self, curviewpoint.cpu().numpy(), self.minimum, self.maximum, num=50000, scale_factor=self.scale_factor)
                        curpoints.append(points)
                        curspeeds.append(speeds)
                        curbounds.append(bounds)          
                    frame_points = torch.cat(curpoints, dim=0)
                    frame_speeds = torch.cat(curspeeds, dim=0)
                    frame_bounds = torch.cat(curbounds, dim=0)
                    frame_epoch = 50

                    #? save occupancy map
                    self.occ_map.save(self.folder+"/occ_map.npz")
                elif self.mode == READ_FROM_DEPTH:
                    # 8 directions for one frame
                    frame_points, frame_speeds, frame_bounds = self.dataset.get_speeds(range(self.frame_idx*8,self.frame_idx*8+8), self.minimum*self.scale_factor, self.maximum*self.scale_factor, num=10000)
                    frame_points /= self.scale_factor
                    frame_bounds /= self.scale_factor
                    frame_epoch = 100
                elif self.mode == READ_FROM_TURTLEBOT:
                    frame_points, frame_speeds, frame_bounds, surf_pc = self.dataset.get_speeds([self.frame_idx], self.minimum*self.scale_factor, self.maximum*self.scale_factor, num=10000)
                    # frame_points, frame_speeds, frame_bounds = self.dataset.get_speeds(range(100), self.minimum*self.scale_factor, self.maximum*self.scale_factor, num=100000)
                    frame_points /= self.scale_factor
                    frame_bounds /= self.scale_factor
                    frame_epoch = 10
                    
                    surf_pc /= self.scale_factor
                    self.all_surf_pc.append(surf_pc)

                    mask = frame_speeds[:, 1] > 0.7
                    self.free_pc.append(frame_points[:,3:][mask])

                    # pc_saved = torch.cat(self.all_surf_pc, dim=0)
                    # np.save("allpc.npy", pc_saved.detach().cpu().numpy())

                    pc_saved = torch.cat(self.free_pc, dim=0)
                    np.save(os.path.join(self.folder,"free_pc.npy"), pc_saved.detach().cpu().numpy())
                elif self.mode == TOY_FB:
                    frame_points, frame_speeds, frame_bounds = generate_sphere_data(limit=0.5, radius=0.2, minimum=self.minimum, maximum=self.maximum,sample_size=1000)
                    frame_epoch = 100


                #! generate data end
                # print("frame_points shape", frame_points.shape)
                # print("frame_speeds shape", frame_speeds.shape)
                frame_data = torch.cat((frame_points.to(self.Params['Device']), frame_speeds.to(self.Params['Device'])), dim=1)
                
                ####################! Core Training #############################
                total_diff, cur_data = self.train_core(frame_epoch, frame_data)



            #! Policy
            # nbv = self.policy_greedy(pre_view, cur_data.detach(), gamma)
            # nbv = self.policy_handcrafted(frame_idx)
            traj_list = None
            if self.mode == EXPLORATION:
                #! Update the occupancy grid
                #? filter frame points with some z range
                valid = (frame_points[:,-1] > -0.1) & (frame_points[:,-1] < 0.1)
                self.occ_map.update(self.cur_view.clone().cpu().numpy(), frame_points[valid].cpu().numpy(), frame_bounds[valid].cpu().numpy())
                coverage = self.occ_map.get_coverage()
                print("Occupancy Grid Coverage:", coverage)
                if coverage > 0.54:
                    pass
                    # break
                traj_list, traj_ind = self.policy_occ(self.cur_view.detach().clone().cpu().numpy(), height=0)
                nbv = Tensor(traj_list[traj_ind])
            elif self.mode == READ_FROM_DEPTH:
                if self.frame_idx + 1 >= len(self.dataset)/8:
                    break
                nbv = Tensor(self.dataset.Ts[self.frame_idx*8][:3, 3]/self.scale_factor)
            elif self.mode == READ_FROM_TURTLEBOT:
                if self.frame_idx >= len(self.dataset):
                    break 
                nbv = Tensor(self.dataset.Ts[self.frame_idx][:3, 3]/self.scale_factor)
            elif self.mode == TOY_FB or True:
                nbv = self.cur_view

            # print("*"*10)
            # print("self.dataset.Ts[0][:3, 3]/self.scale_factor", self.dataset.Ts[0][:3, 3]/self.scale_factor)
            print("nbv", nbv)
            print("curview", self.cur_view)
            traj = self.predict_trajectory2(self.cur_view.detach().clone().cpu().numpy(), nbv.detach().clone().cpu().numpy())
            if self.mode == EXPLORATION:
                traj = traj_list[:traj_ind+1]
            if self.trajectory is None:
                self.trajectory = traj
            else:
                self.trajectory = np.concatenate([self.trajectory, traj], axis=0) 
            print("traj is:", traj)
            
            
            #? ******************SAVINGS start*******************
            save_traj = False 
            if save_traj:
                np.save(self.folder+"/traj"+"_"+str(self.epoch)+".npy", self.trajectory)



            # if (self.epoch % self.Params['Training']['Save Every * Epoch'] == 0) or (self.epoch == self.Params['Training']['Number of Epochs']) or (self.epoch == 1):
            # print("cur view",self.cur_view)
            # print("nbv",nbv)
            
            camera_matrix = None
            if self.mode == READ_FROM_TURTLEBOT:
                camera_matrix = self.dataset[self.frame_idx]["T"]
            
            if self.mode == TOY_FB:
                self.plot2d(self.cur_view, nbv, self.epoch, total_diff.item(),self.alpha, cur_data[:,:6].clone().cpu().numpy(), camera_matrix, traj_list)
            else:
                self.plot2(self.cur_view, nbv, self.epoch, total_diff.item(),self.alpha)

            
            # self.fourplot(epoch, FRAMES[:frame_idx+1], total_diff.item(), alpha)
            with torch.no_grad():
                self.save(epoch=self.epoch, val_loss=total_diff)

            #? ******************SAVINGS end*******************
            
            self.frame_idx += 1
            if self.frame_idx >= self.camera_steps:
                break
            
            self.cur_view = nbv

        if False:
            print("Exploration is done. Finetuning...")
            self.plot2(self.cur_view, nbv, self.epoch, total_diff.item(),self.alpha)
            #TODO: train for another 1000 epochs for finetuning
            # 1. data is all framedata
            prev_frame_idx = torch.randperm(self.frame_idx).tolist()[:self.frame_buffer_size-1]
            #! mix data so that the start and end points are from different frames
            prev_framedata = self.all_framedata[prev_frame_idx]
            chosen_percent = 1.0
            # print("chosen points shape:", chosen_points.shape)
            chosen_framedata = torch.cat((prev_framedata, frame_data.unsqueeze(0)), dim=0).view(-1, 8)

            chosen_points = chosen_framedata[:,:6]
            chosen_speeds = chosen_framedata[:,6:]
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
            fine_data = torch.cat((global_points_combination, global_speeds_combination), dim=1)

            # 2. core training
            fine_epoch = 1000
            total_diff, cur_data = self.train_core(fine_epoch, fine_data, False)

            # 3. save a final plot #TODO: may need some other plots
            self.plot2(self.cur_view, nbv, self.epoch, total_diff.item(),self.alpha)
            with torch.no_grad():
                self.save(epoch=self.epoch, val_loss=total_diff)
            # self.plot2(self.cur_view, nbv, self.epoch, total_diff.item(),self.alpha, cur_data[:,:6].clone().cpu().numpy(), camera_matrix)

    def train_core(self, epoch, frame_data, is_one_frame=True):
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1.0
        step = -2000.0/4000.0
        step = 0
        tt =time.time()

        if is_one_frame:
            if self.all_framedata is None:
                self.all_framedata = frame_data.unsqueeze(0)
            else:
                self.all_framedata = torch.cat((self.all_framedata, frame_data.unsqueeze(0)), dim=0)


        #! mix data so that the start and end points are from different frames
        if is_one_frame:
            cur_data = self.randomize_data(frame_data)
            # print("cur_data:", cur_data.shape)
        else:
            num_samples = frame_data.shape[0]
            start_indices = torch.randperm(num_samples)
            end_indices = torch.randperm(num_samples)
            start_points = frame_data[start_indices, :3]
            start_speeds = frame_data[start_indices, 6]
            end_points = frame_data[end_indices, 3:6]
            end_speeds = frame_data[end_indices, 7]
            global_points_combination = torch.cat((start_points, end_points), dim=1)
            global_speeds_combination = torch.cat((start_speeds[:, None], end_speeds[:, None]), dim=1)
            cur_data = torch.cat((global_points_combination, global_speeds_combination), dim=1)
            
        if self.mode == READ_FROM_COOKED_DATA: #? use all framedata and mix it
            # chosen_percent = 1.0
            # # print("chosen points shape:", chosen_points.shape)
            # chosen_start_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            # chosen_end_index = torch.randperm(chosen_points.shape[0])[:int(chosen_points.shape[0]*chosen_percent)]
            # # print("chosen points indice shape:", chosen_start_index.shape)
            # chosen_start_points = chosen_points[chosen_start_index][:, :self.dim]
            # chosen_start_speeds = chosen_speeds[chosen_start_index][:, 0]
            # chosen_end_points = chosen_points[chosen_end_index][:, self.dim:2*self.dim]
            # chosen_end_speeds = chosen_speeds[chosen_end_index][:, 1]
            # # print("chosen start points shape:", chosen_start_points.shape)
            # global_points_combination = torch.cat((chosen_start_points, chosen_end_points), dim=1)
            # global_speeds_combination = torch.cat((chosen_start_speeds[:, None], chosen_end_speeds[:, None]), dim=1)

            # all_points = torch.cat((local_points_combination, global_points_combination), dim=0)
            # all_speeds = torch.cat((local_speeds_combination, global_speeds_combination), dim=0)

            # cur_data = torch.cat((all_points, all_speeds), dim=1)

            dataloader = FastTensorDataLoader(cur_data, batch_size=int(self.Params['Training']['Batch Size']), 
            shuffle=True)
        else: #? use current framedata continual learning
            dataloader = FastTensorDataLoader(cur_data, batch_size=int(self.Params['Training']['Batch Size']), shuffle=True)


        # frame_epoch = 10000 if isOffline else 50
        frame_epoch = epoch
        start_time = time.time()
        for e in range(frame_epoch):
            self.epoch += 1
            total_train_loss = 0
            total_diff=0

            # alpha = min(max(0.85,0.85+0.5*step),0.9)
            # alpha = 1.10#1.05
            

            step+=1.0/4000/((int)(self.epoch/4000)+1.) * 2.0
            gamma=0.001#max((4000.0-epoch)/4000.0/20,0.001)

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
        end_time = time.time()
        if is_one_frame:
            duration = end_time-start_time
            print(f"running in {duration} secs")
            self.timer.append(duration)
            np.save("gib6_time.npy",np.array(self.timer))
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
     
    def Gradient3(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
       
        #Xp.requires_grad_()
        
        #tau, dtau, coords = self.network.out_grad(Xp)

        tau, Xp = self.network.out(Xp)
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

    def TravelTimes2(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.network.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT
    
    def Tau2(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.network.out(Xp)

        return tau

    def Speed2(self, Xp):
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

        T1    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT1, DT1)
        T2    = 4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        
        
        S = (T1+T2+LogTau4)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3, LogTau,LogTau2,LogTau3,LogTau4,DT1,S
        return Ypred


    def plot2(self, src, tar, epoch, total_train_loss, alpha, cur_points=None, camera_matrix=None, traj_list = None):
        limit = 1
        xmin = [-0.5, -0.5]
        xmax = [0.5, 0.5]
        if self.mode == READ_FROM_TURTLEBOT:
            xmin = [-1.5, -1.5]
            xmax = [1.5, 1.5]
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
        
        tt = self.TravelTimes2(XP)
        ss = self.Speed2(XP)#*5
        tau = self.Tau2(XP)
        
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

        # draw occupancy grid
        if self.mode in [EXPLORATION, TURTLEBOT_EXPLORATION]:
            pathname = self.folder+"/plots"+str(epoch)+"_"+"occ"+".png"
            cur_small_loc = src.cpu().numpy()
            tar_small_loc = tar.cpu().numpy()
            self.occ_map.plot(cur_small_loc, tar_small_loc, path=pathname)
            


        # draw four more plots
        if False:
            height = 1.2
            prelocs = [[-7, 4, height], [-7, 1, height], [-1, 1, height], [-3, 4.7, height]]
            for i,loc in enumerate(prelocs):

                Xsrc = np.array(loc)
                Xsrc[0] += 4.2529825
                Xsrc[1] -= 2.682928
                Xsrc[2] -= 1.1388535
                Xsrc /= 9.757375
                
                XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
                XP[:,:self.dim] = Xsrc
                XP[:,self.dim:] = Xsrc
                #XP=XP/scale
                XP[:,self.dim+0]  = X.flatten()
                XP[:,self.dim+1]  = Y.flatten() #! this allow to change from y to z
                XP = Variable(Tensor(XP)).to(self.Params['Device'])
                
                tt = self.TravelTimes2(XP)
                ss = self.Speed2(XP)#*5
                tau = self.Tau2(XP)
                
                TT = tt.to('cpu').data.numpy().reshape(X.shape)
                V  = ss.to('cpu').data.numpy().reshape(X.shape)
                TAU = tau.to('cpu').data.numpy().reshape(X.shape)

                fig = plt.figure()

                ax = fig.add_subplot(111)
                # ax.invert_xaxis()
                # ax.invert_yaxis()
                quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=1)
                
                ax.contour(X,Y,TT,np.arange(0,2,0.02), cmap='bone', linewidths=0.3)#0.25
                plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
                plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(i)+".png",bbox_inches='tight')
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

    def plot2d(self, src, tar, epoch, total_train_loss, alpha, cur_points=None, camera_matrix=None, traj_list = None):
        limit = 1
        xmin = [-1, -1]
        xmax = [1, 1] 
        xmin = [-0.5, -0.5]
        xmax = [0.5, 0.5] 
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
        XP[:,self.dim+1]  = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.Params['Device'])
        
        tt = self.TravelTimes2(XP)
        ss = self.Speed2(XP)#*5
        tau = self.Tau2(XP)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        # ax.invert_xaxis()
        # ax.invert_yaxis()
        quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=1)

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

        # draw occupancy grid
        if self.mode in [EXPLORATION, TURTLEBOT_EXPLORATION]:
            pathname = self.folder+"/plots"+str(epoch)+"_"+"occ"+".png"
            cur_small_loc = src.cpu().numpy()
            tar_small_loc = tar.cpu().numpy()
            self.occ_map.plot(cur_small_loc, tar_small_loc, path=pathname)

        if True: #? Draw ground truth 
            gt_V = np.zeros_like(V)
            gt_distance = np.linalg.norm(np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1), axis=1) - 0.25
            gt_V_unshaped = np.clip(gt_distance, self.minimum, self.maximum)/self.maximum
            gt_V = gt_V_unshaped.reshape(X.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            quad1 = ax.pcolormesh(X,Y,gt_V,vmin=0,vmax=1)
            plt.colorbar(quad1,ax=ax, pad=0.1, label='GT Velocity')
            plt.savefig(self.folder+"/plots_gt.png",bbox_inches='tight')
            plt.close(fig)
        
        if True: #?visualize input data
            alldata = self.all_framedata.detach().cpu().numpy().reshape(-1, 6)
            start_points = alldata[:,:2]
            end_points = alldata[:,2:4]
            start_speeds = alldata[:,4]
            end_speeds = alldata[:,5]
            # gt_V = np.zeros_like(V)
            # gt_distance = np.linalg.norm(np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1), axis=1) - 0.5
            # gt_V_unshaped = np.clip(gt_distance, self.minimum, self.maximum)/self.maximum
            # gt_V = gt_V_unshaped.reshape(X.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.xlim(-0.5, 0.5)
            plt.ylim(-0.5, 0.5)
            plt.scatter(start_points[:, 0], start_points[:, 1], c=start_speeds, cmap='viridis')
            # plt.colorbar
            # quad1 = ax.pcolormesh(X,Y,gt_V,vmin=0,vmax=1)
            plt.colorbar(quad1,ax=ax, pad=0.1, label='start Velocity')
            plt.savefig(self.folder+"/plots_input_start"+str(epoch)+".png",bbox_inches='tight')
            plt.close(fig)


            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.scatter(end_points[:, 0], end_points[:, 1], c=end_speeds, cmap='viridis')
            plt.xlim(-0.5, 0.5)
            plt.ylim(-0.5, 0.5)
            # plt.colorbar
            # quad1 = ax.pcolormesh(X,Y,gt_V,vmin=0,vmax=1)
            plt.colorbar(quad1,ax=ax, pad=0.1, label='end Velocity')
            plt.savefig(self.folder+"/plots_input_end"+str(epoch)+".png",bbox_inches='tight')
            plt.close(fig)


            

    def predict_trajectory2(self, Xsrc, Xtar, step_size=0.03, tol=0.03):
        Xsrc = Tensor(Xsrc)
        Xtar = Tensor(Xtar)
        XP_traj= torch.cat((Xsrc,Xtar))
        dis= torch.norm(XP_traj[:self.dim]-XP_traj[self.dim:])
        point_start = []
        point_goal = []
        iter = 0
        while dis > tol:
            gradient = self.Gradient3(XP_traj[None,:].clone())
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
            gradient = self.Gradient3(XP_traj[None,:].clone())
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
    
    def policy_occ(self, current_location, height):
        # curtargetblock = self.occ_map.get_target_block()
        # if not curtargetblock:
        #     return Variable(Tensor([0, 0, height])).cpu()
        # curtargetloc = self.occ_map.get_block_center(curtargetblock)
        possible_locs = self.occ_map.get_block_centers()
        if False:
            #? find the closest one
            curxy = current_location[:2]
            min_dist = 100
            curtargetloc = None
            block_idx = -1
            for k,loc in enumerate(possible_locs):
                loc = mesh_small_to_large(loc)
                dist = np.linalg.norm(curxy - loc)
                if dist < min_dist:
                    min_dist = dist
                    curtargetloc = loc
                    block_idx = k
            if curtargetloc is None:
                print("Warning: current location is not in the occupancy grid !!!")
                return Variable(Tensor([0, 0, height])).cpu()
        else:
            #? find the shortest path one
            """
            XP       = np.zeros((len(X.flatten()),2*self.dim))#*((xmax[dims_n]-xmin[dims_n])/2 +xmin[dims_n])
            XP[:,:self.dim] = Xsrc
            XP[:,self.dim:] = Xsrc
            #XP=XP/scale
            XP[:,self.dim+0]  = X.flatten()
            XP[:,self.dim+1]  = Y.flatten() #! this allow to change from y to z
            XP = Variable(Tensor(XP)).to(self.Params['Device'])
            
            tt = self.TravelTimes2(XP)
            """
            XP = np.zeros((len(possible_locs),2*self.dim))

            #?: change to small locations
            XP[:,:self.dim] = current_location
            XP[:,self.dim:self.dim+2] = possible_locs
            XP[:,self.dim+2] = height
            # print("XP", XP)
            XP = Variable(Tensor(XP)).to(self.Params['Device'])
            tt = self.TravelTimes2(XP)
            tt = tt.to('cpu').data.numpy()
            small_idx = np.argmin(tt)
            curtargetloc_small = possible_locs[small_idx]  
        
        print("current location :", current_location)
        print("target location :", curtargetloc_small)
        print("block idx:", self.occ_map.block_stack[small_idx])

        #！ set a step size
        curtargetloc_small = np.concatenate([curtargetloc_small, [height]])
        curtargetloc_small_tensor = torch.from_numpy(curtargetloc_small).float()
        
        curloc_tensor = torch.from_numpy(current_location).float()

        traj_list = self.predict_trajectory2(curloc_tensor, curtargetloc_small_tensor, step_size=0.02)

        # find the largest point that is smaller than the step size
        if True:
            #? accumulate the distance to fit the step size
            step_size = 0.05
            index = 0
            curtargetloc = traj_list[-1]
            accum_dis = 0
            while accum_dis < step_size:
                curtargetloc = traj_list[index]
                accum_dis += np.linalg.norm(traj_list[index+1][:2] - traj_list[index][:2])
                index += 1
                if index >= len(traj_list)-1:
                    break
            print("accumulated distance:", accum_dis, "at traj_index:", index)
            print("Target location after gradient descent:", curtargetloc)
            print("Length of traj_list:", len(traj_list))

        #! should return a list of trajectory, and index of the end traj
        # return Variable(Tensor(curtargetloc)).cpu()
        return traj_list, index - 1
    
    
