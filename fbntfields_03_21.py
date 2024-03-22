import torch 
import numpy as np
import matplotlib.pyplot as plt
# import torch.nn as nn
from nn import NN
from fb_data import generate_sphere_data, FastTensorDataLoader
import time
import pickle
import random
import datetime
import os

minimum = 0.01 
maximum = 0.1

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

def plot_subdomain_windows(decomposition_init_kwargs, device='cpu'):
    # Prepare the input range (e.g., 0 to 1)
    t_vals = torch.linspace(-0.5, 0.5, 1000, device=device)
    
    plt.figure(figsize=(10, 6))
    
    # subdomain_xs = torch.linspace(*decomposition_init_kwargs['unnorm'], decomposition_init_kwargs['subdomain_xs'].size)
    subdomain_xs = torch.tensor(decomposition_init_kwargs['subdomain_xs'], device=device)
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
    plt.savefig('fbntfields_subdomain_windows.png')
    plt.close()

class NTSubNN(torch.nn.Module):
    def __init__(self, dim, mid, width, device):
        super(NTSubNN, self).__init__()
        self.dim = dim
        self.device = device
        self.mid = mid
        self.width = width
        self.B = 0.05 * torch.normal(0,1,size=(128,self.dim)) #0.5
        self.network = NN(self.device, self.dim, self.B)
        self.network.apply(self.network.init_weights)
        self.network.to(self.device)

    def forward(self, x):
        norm_x = norm(self.mid, self.width, x)
        tau, Xp = self.network.out(norm_x)
        # mu = 0
        # sd = 1 
        # unnorm_tau = unnorm(mu, sd, tau)
        unnorm_tau = tau
        #! use start x coordinate to get the window
        window = cosine_window(x[:, 0, None], self.mid, self.width, self.device)
        return window * unnorm_tau, window, tau  
        

class NTNN(torch.nn.Module):
    def __init__(self, decomposition_init_kwargs, dim, dataloader, device='cpu'):
        super(NTNN, self).__init__()
        self.decomposition_init_kwargs = decomposition_init_kwargs
        self.device = device
        self.dim = dim
        self.dataloader = dataloader
        self.subnets = torch.nn.ModuleList([NTSubNN(self.dim, mid, width, device) for mid, width in zip(decomposition_init_kwargs['subdomain_xs'], decomposition_init_kwargs['subdomain_ws'])])

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.1)

        self.total_train_loss = []
        self.total_val_loss = []
        self.epoch = 0
        self.frame_idx = 0
        self.trajectory = None
        self.prev_state_queue = []
        self.prev_optimizer_queue = []

        current_time = datetime.datetime.now() - datetime.timedelta(hours=5)
        self.folder = "Experiments/"+current_time.strftime("%m_%d_%H_%M")
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)


    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        outputs = []
        windows = []
        for subnet in self.subnets:
            val, window, raw_val = subnet(x)
            outputs.append(val)
            windows.append(window)
        outputs = torch.stack(outputs, dim=0)
        outputs = torch.sum(outputs, dim=0)
        windows = torch.stack(windows, dim=0)
        weight = torch.sum(windows, dim=0)
        outputs = outputs/weight 
        return outputs, x
    
    def gradient(self, y, x, create_graph=True):
        grad_y = torch.ones_like(y, requires_grad=False)
        grad_x = torch.autograd.grad(y, x, grad_y, retain_graph=True, create_graph=create_graph)[0]
        return grad_x

    def train(self, epoch):
        beta = 1.0
        prev_diff = 1.0
        current_diff = 1.0
        gamma = 0.001
        self.alpha = 1.0

        current_state = pickle.loads(pickle.dumps(self.subnets.state_dict()))
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
            for i, data in enumerate(self.dataloader, 0):#train_loader_wei,dataloader
                #print('----------------- Epoch {} - Batch {} --------------------'.format(epoch,i))
                if i>5:
                    break
                t0 = time.time()

                data=data[0].to(self.device)
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
    
            if (diff_ratio < 1.2 and diff_ratio > 0 or epoch < 10):#1.5
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
        
        return total_diff 


    def Loss(self, points, Yobs, beta, gamma):
        start=time.time()
        tau, Xp = self(points) #todo: change to out
        # Xp = points.clone().detach().requires_grad_(True)
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
    

    def TravelTimes2(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.device))
        
        tau, coords = self(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim] 
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT
    
    def Tau2(self, Xp):
        Xp = Xp.to(torch.device(self.device))
     
        tau, coords = self(Xp)

        return tau

    def Speed2(self, Xp):
        Xp = Xp.to(torch.device(self.device))

        tau, Xp = self(Xp)
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
    

    def plot2d(self, src, tar, epoch, total_train_loss, alpha, traj_list = None):
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
        # XP = Variable(Tensor(XP)).to(self.device)
        XP = torch.tensor(XP, dtype=torch.float32, device=self.device)
        
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
        # ax.plot(self.trajectory[:, 0], self.trajectory[:, 1], color='red', marker='o', markersize=0.8, linestyle='-', linewidth=1)
        

        ax.contour(X,Y,TT,np.arange(0,5,0.02), cmap='bone', linewidths=0.3)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.folder+"/plots"+str(epoch)+"_"+str(alpha)+"_"+str(round(total_train_loss,4))+"_0.png",bbox_inches='tight')

        plt.close(fig)

        # # draw occupancy grid
        # if self.mode in [EXPLORATION, TURTLEBOT_EXPLORATION]:
        #     pathname = self.folder+"/plots"+str(epoch)+"_"+"occ"+".png"
        #     cur_small_loc = src.cpu().numpy()
        #     tar_small_loc = tar.cpu().numpy()
        #     self.occ_map.plot(cur_small_loc, tar_small_loc, path=pathname)

        if True: #? Draw ground truth 
            gt_V = np.zeros_like(V)
            gt_distance = np.linalg.norm(np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1), axis=1) - 0.25
            gt_V_unshaped = np.clip(gt_distance, minimum, maximum)/maximum
            gt_V = gt_V_unshaped.reshape(X.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            quad1 = ax.pcolormesh(X,Y,gt_V,vmin=0,vmax=1)
            plt.colorbar(quad1,ax=ax, pad=0.1, label='GT Velocity')
            plt.savefig(self.folder+"/plots_gt.png",bbox_inches='tight')
            plt.close(fig)
        
        if True: #?visualize input data
            # alldata = self.all_framedata.detach().cpu().numpy().reshape(-1, 6)
            alldata = self.dataloader.tensors[0].detach().cpu().numpy().reshape(-1, 6)[:1000]
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


        
        


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decomposition_init_kwargs = {
        'subdomain_xs': np.linspace(-0.5, 0.5, 5),  # 15 equally spaced subdomains
        'subdomain_ws': 0.5 * np.ones((5,)),  # with widths of 0.15
        'unnorm': (0., 1.),  # define unnormalisation of the subdomain networks
    }
    # beta = 1.0
    # prev_diff = 1.0
    # current_diff = 1.0
    # step = -2000.0/4000.0
    # step = 0

    plot_subdomain_windows(decomposition_init_kwargs, device)


    #! Generate sphere data
    points, speeds, bounds = generate_sphere_data(limit=0.5, radius=0.2, minimum=minimum, maximum=maximum, sample_size=100000)

    data = torch.cat((points, speeds), dim=1)
    dataloader = FastTensorDataLoader(data, batch_size=20000, shuffle=True)

    model = NTNN(decomposition_init_kwargs, dim=2, dataloader=dataloader, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    

    start_time = time.time()
    for epoch in range(20001):
        loss = model.train(epoch)

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/20000], Loss: {loss.item():.4f}')
            # add some plot if needed
            
            end_time = time.time()
            print(f"Time elapsed: {end_time-start_time} seconds")
            start_time = end_time

            #! Visualize the model
            src = torch.tensor([-0.1, -0.4])
            model.plot2d(src, None, epoch, loss.item(), 1.0)



