import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

USE_FILTER = False

def lip_norm(w):
    #y = x@w.T+b
    absrowsum = torch.sqrt(torch.sum (  w**2 , dim =1))
    #print(absrowsum.shape)
    scale = torch.clamp (1 / absrowsum ,max=1)#.squeeze()
    #scale = self.actout(torch.exp(100*c[0,0]**2) / absrowsum )
    #scale = c[0,0]**2/absrowsum#torch.clamp (torch.exp(100*c[0,0]) / absrowsum ,max=1)
    #print(scale.shape)
    #print(w.shape)
    return w * scale.unsqueeze(1)

def l1_distance(f_start, f_goal):
    return torch.sum(torch.abs(f_start - f_goal), dim=1)

def l2_distance(f_start, f_goal):
    return torch.sqrt(torch.sum((f_start - f_goal)**2, dim=1))

def cosine_distance(f_start, f_goal):
    f_start_norm = F.normalize(f_start, p=2, dim=1)
    f_goal_norm = F.normalize(f_goal, p=2, dim=1)
    cosine_similarity = torch.sum(f_start_norm * f_goal_norm, dim=1)
    return 1 - cosine_similarity

def linf_distance(f_start, f_goal):
    return torch.max(torch.abs(f_start - f_goal), dim=1)[0]

# def lip_norm(self, w):
#         #y = x@w.T+b
#         absrowsum = torch.sqrt(torch.sum (  w**2 , dim =1))
#         #print(absrowsum.shape)
#         scale = torch.clamp (1 / absrowsum ,max=1)#.squeeze()
#         #scale = self.actout(torch.exp(100*c[0,0]**2) / absrowsum )
#         #scale = c[0,0]**2/absrowsum#torch.clamp (torch.exp(100*c[0,0]) / absrowsum ,max=1)
#         #print(scale.shape)
#         #print(w.shape)
#         return w * scale.unsqueeze(1) #[: , None ]

def normalization(w, softplus_ci):
    absrowsum = torch.sum(torch.abs(w), dim=1)
    scale = torch.min(torch.tensor(1.0), softplus_ci / absrowsum)
    return w * scale.unsqueeze(1)

class SubNN(nn.Module):
    def __init__(self):
        super(SubNN, self).__init__()

    # @staticmethod
    # def init_network(nl, input_size, h_size):
    #     encoder = torch.nn.ModuleList()
    #     encoder1 = torch.nn.ModuleList()
    #     encoder.append(Linear(2*input_size, h_size))
    #     encoder1.append(Linear(h_size, h_size))

    #     for i in range(nl):
    #         encoder.append(Linear(h_size, h_size))
    #         encoder1.append(Linear(h_size, h_size))
    #     encoder.append(Linear(h_size, h_size))
    #     return encoder, encoder1
    
    @staticmethod
    def init_network(layer_sizes, input_size):
        encoder = torch.nn.ModuleList()
        # layer_sizes = [64, 64, 64, 64, 256]
        encoder.append(Linear(2*input_size, layer_sizes[0]))

        for i in range(0, len(layer_sizes)-1):
            encoder.append(Linear(layer_sizes[i], layer_sizes[i+1]))
        # encoder.append(Linear(h_size, 256))
        return encoder
    


    @staticmethod
    def encoder_out(submodel, coords):
        self = submodel
        use_lip = self.config['model']['use_lipschitz']
        use_1DNorm = self.config['model']['use_1DNorm']
        if not USE_FILTER:
            # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

            x0 = coords[:,:self.dim]
            x1 = coords[:,self.dim:]
            x = torch.vstack((x0,x1))
            x = self.input_mapping(x)
        else:
            # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
            # x0 = coords[:,:self.dim]
            # x1 = coords[:,self.dim:]
            # x = torch.vstack((x0,x1))
            x = self.input_mapping(coords)

        if use_lip:
            lip_const = self.config['model']['lip_const']
            w = self.encoder[0].weight
            b = self.encoder[0].bias

            #c = self.encoder_lip[0].weight
            w = lip_norm(w)

            y = x@w.T+b
            
            if use_1DNorm:
                y = y.unsqueeze(1)
                y = self.norm_layers[0](y)
                y = y.squeeze(1)
            x = torch.sin(lip_const*y)

            for ii in range(1,self.nl+1):
                w = self.encoder[ii].weight
                b = self.encoder[ii].bias

                #c = self.encoder_lip[ii].weight
                w = lip_norm(w)

                y = x@w.T+b
                if use_1DNorm:
                    y = y.unsqueeze(1)
                    y = self.norm_layers[ii](y)
                    y = y.squeeze(1)
                x  = torch.sin(lip_const*y)

            # x = self.encoder[-1](x)
        else:
            x = self.encoder[0](x)
            if use_1DNorm:
                x = x.unsqueeze(1)
                x = self.norm_layers[0](x)
                x = x.squeeze(1)
            x = torch.sin(x)

            for ii in range(1,self.nl):
                # x = torch.sin(self.encoder[ii](x))
                x = self.encoder[ii](x)
                if use_1DNorm:
                    x = x.unsqueeze(1)
                    x = self.norm_layers[ii](x)
                    x = x.squeeze(1)
                x = torch.sin(x)

            x = self.encoder[-1](x)
        return x, coords
    
    @staticmethod
    def encoder_out_new(submodel, coords):
        self = submodel
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        use_lip = self.config['model']['use_lipschitz']
        use_1DNorm = self.config['model']['use_1DNorm']
        # x0 = coords[:,:self.dim]
        # x1 = coords[:,self.dim:]
        # x = torch.vstack((x0,x1))
        x = self.input_mapping(coords)

        if use_lip:
            lip_const = self.config['model']['lip_const']
            w = self.encoder[0].weight
            b = self.encoder[0].bias

            #c = self.encoder_lip[0].weight
            w = lip_norm(w)

            y = x@w.T+b
            
            if use_1DNorm:
                y = y.unsqueeze(1)
                y = self.norm_layers[0](y)
                y = y.squeeze(1)
            x = torch.sin(lip_const*y)

            for ii in range(1,self.nl+1):
                w = self.encoder[ii].weight
                b = self.encoder[ii].bias

                #c = self.encoder_lip[ii].weight
                w = lip_norm(w)

                y = x@w.T+b
                if use_1DNorm:
                    y = y.unsqueeze(1)
                    y = self.norm_layers[ii](y)
                    y = y.squeeze(1)
                x  = torch.sin(lip_const*y)

            # x = self.encoder[-1](x)
        else:
            x = self.encoder[0](x)
            if use_1DNorm:
                x = x.unsqueeze(1)
                x = self.norm_layers[0](x)
                x = x.squeeze(1)
            x = torch.sin(x)

            for ii in range(1,self.nl):
                # x = torch.sin(self.encoder[ii](x))
                x = self.encoder[ii](x)
                if use_1DNorm:
                    x = x.unsqueeze(1)
                    x = self.norm_layers[ii](x)
                    x = x.squeeze(1)
                x = torch.sin(x)

            x = self.encoder[-1](x)
        return x, coords

    @staticmethod
    def encoder_out_nowork(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))
        
        x = self.input_mapping(x)
        w = self.encoder[0].weight
        b = self.encoder[0].bias

        #c = self.encoder_lip[0].weight
        # w = lip_norm(w)
        self.softplus_ci = torch.tensor(10.0)
        w = normalization(w, self.softplus_ci)

        y = x@w.T+b
        x = torch.sin(y)

        for ii in range(1,self.nl):
            w = self.encoder[ii].weight
            b = self.encoder[ii].bias

            #c = self.encoder_lip[ii].weight
            # w = lip_norm(w)
            w = normalization(w, self.softplus_ci)

            y = x@w.T+b
            x  = torch.sin(y)

        x = self.encoder[-1](x)
        return x, coords


    @staticmethod
    def encoder_out_origin(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))
        
        x = self.input_mapping(x)
        x = torch.sin(self.encoder[0](x))

        for ii in range(1,self.nl):
            x = torch.sin(self.encoder[ii](x))

        x = self.encoder[-1](x)
        return x, coords

    
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
    
    # @staticmethod
    # def sym_op(normalized_features, Xp):
    #     x0 = normalized_features[:Xp.shape[0],:]#.unsqueeze(1)
    #     x1 = normalized_features[Xp.shape[0]:,:]#.unsqueeze(1)
    #     x = x0-x1
    #     x = torch.sqrt((x0-x1)**2+0.00001)
    #     x = torch.logsumexp(10*x, 1).unsqueeze(1)/10
    #     return x, Xp

    @staticmethod
    def sym_op(model, normalized_features, Xp):
        x0 = normalized_features[:Xp.shape[0],:]#.unsqueeze(1)
        x1 = normalized_features[Xp.shape[0]:,:]#.unsqueeze(1)


        sym_op_type = model.config['model']['sym_op_type']
        if sym_op_type == "Sphere":
            scale0 = torch.sqrt(torch.sum (  ( x0**2 ) , dim =1)).unsqueeze(1) 
            x0 = x0/scale0

            scale1 = torch.sqrt(torch.sum (  ( x1**2 ) , dim =1)).unsqueeze(1) 
            x1 = x1/scale1

            x = x0*x1

            x = torch.sum(x,dim=1).unsqueeze(1)
            sym_div_hype = model.config['model']['sym_div_hype']
            x = sym_div_hype * torch.acos(x-0.000001)/(scale0*scale1)
            # x = torch.acos(x-0.000001)
        elif sym_op_type == "L1":
            x = torch.sqrt((x0-x1)**2+1e-6)
            x = x.view(x.shape[0], -1, 16)
            x = torch.logsumexp(10*x, -1)
            # x =  x.unsqueeze(1)
            x = 0.1 * torch.sum(x, dim=1).unsqueeze(1)
            #? this const smaller, lighter   
        elif sym_op_type == "L2_old":
            x = torch.sqrt(torch.sum((x0 - x1) ** 2, dim=1) + 1e-6)
            x = 3 * x.unsqueeze(1)
        elif sym_op_type == "L2":
            x = torch.sqrt((x0-x1)**2+1e-6)
            x = x.view(x.shape[0], -1, 16)
            # x = torch.sqrt(torch.sum((x0 - x1) ** 2, dim=1) + 1e-6)
            x = torch.logsumexp(1*x, -1)
            x = 1 * torch.sum(x, dim=1).unsqueeze(1)
        elif sym_op_type == "L1_sqrt":
            # diff = torch.abs(x0 - x1)
            diff = torch.sqrt((x0-x1)**2+1e-6)
            x = torch.logsumexp(10*diff, -1)
            x = 1 * x.unsqueeze(1)
        elif sym_op_type == "L1_new":
            diff = torch.abs(x0 - x1)
            # diff = torch.sqrt((x0-x1)**2+1e-6)
            x = torch.logsumexp(10*diff, -1)
            x = 1 * x.unsqueeze(1)
        # x = 1000*torch.acos(x-0.000001)/(scale0*scale1) #250
        # x = torch.acos(x-0.000001)*(scale0*scale1)/50 #250
        return x, Xp

    # @staticmethod
    # def sym_op(model, normalized_features, Xp):
    #     x0 = normalized_features[:Xp.shape[0],:]#.unsqueeze(1)
    #     x1 = normalized_features[Xp.shape[0]:,:]#.unsqueeze(1)

    #     x = torch.sqrt((x0-x1)**2+1e-6)
    #     x = x.view(x.shape[0], -1, 16)
    #     x = torch.logsumexp(10*x, -1)
    #     # x =  x.unsqueeze(1)
    #     x = 0.15 * torch.sum(x, dim=1).unsqueeze(1)
    #     #? this const smaller, lighter

    #     # scale0 = torch.sqrt(torch.sum (  ( x0**2 ) , dim =1)).unsqueeze(1) 
    #     # x0 = x0/scale0

    #     # scale1 = torch.sqrt(torch.sum (  ( x1**2 ) , dim =1)).unsqueeze(1) 
    #     # x1 = x1/scale1

    #     # x = x0*x1

    #     # x = torch.sum(x,dim=1).unsqueeze(1)
    #     # sym_div_hype = model.config['model']['sym_div_hype']
    #     # x =  torch.acos(x-0.000001)
    #     # #! use L1 norm
    #     # x = l1_distance(x0, x1)
    #     return x, Xp
    
    @staticmethod
    def Loss(model, points, Yobs, beta):
        self = model
        learned_regions = self.learned_regions
        active_regions = self.active_regions
        region = set(learned_regions).union(set(active_regions))
        # region = self.all_regions
        tau, Xp = self.out(points, region)
        dtau = self.gradient(tau, Xp)
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        # D_norm = torch.sqrt(torch.einsum('ij,ij->i', D, D))
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
        # T00 = tau[:,0]**2

        epsilon = 1e-6
        # epsilon = 0

        Ypred0 = torch.sqrt(S0+epsilon)
        Ypred1 = torch.sqrt(S1+epsilon)
        sq_Yobs0 = Yobs[:,0]
        sq_Yobs1 = Yobs[:,1]
        l0 = sq_Yobs0*(Ypred0)
        l1 = sq_Yobs1*(Ypred1)
        loss0 = (torch.sqrt(l0+epsilon)-1)**2
        loss1 = (torch.sqrt(l1+epsilon)-1)**2
        diff = loss0 + loss1
        
        #
        # loss_n = torch.sum(diff*torch.exp(-0.1*tau[:, 0]))/Yobs.shape[0]
        # loss_n = torch.sum((loss0 + loss1)*torch.exp(-0.5*tau[:,0]))/Yobs.shape[0]
        # loss_n = torch.sum((loss0 + loss1)*torch.exp(-2.0*tau[:,0]))/Yobs.shape[0]
        loss_n_hype = model.config['model']['loss_n_hype']
        loss_n = torch.sum((loss0 + loss1)*torch.exp(loss_n_hype*tau[:,0]))/Yobs.shape[0] #-0.4
        # loss_n = torch.sum((loss0+loss1))


        
        loss = beta*loss_n #+ 1e-4*(reg_tau)

        del points, Yobs, tau, dtau, D, DT0, DT1, S0, S1, Ypred0, Ypred1, sq_Yobs0, sq_Yobs1, l0, l1, loss0, loss1

        return loss, loss_n, diff

    @staticmethod
    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        region = self.all_regions
        tau, coords = self.out(Xp, region)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        D_norm = torch.sqrt(torch.einsum('ij,ij->i', D, D))

        TT = tau[:,0]

        del Xp, tau, D, D_norm
        return TT
    
    @staticmethod
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
        region = self.all_regions 
        tau, coords = self.out(Xp, region)

        return tau


    @staticmethod
    def Speed(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
        region = self.all_regions
        # tau
        tau, Xp = self.out(Xp, region)
        # dtau
        dtau = self.gradient(tau, Xp)
        # (s-g)
        # D = Xp[:,self.dim:]-Xp[:,:self.dim]
        # D_norm = torch.sqrt(torch.einsum('ij,ij->i', D, D))

        DT1 = dtau[:,self.dim:]
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
        
        Ypred = 1/torch.sqrt(S1)

        
        del Xp, tau, dtau
        return Ypred



