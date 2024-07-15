import torch
import torch.nn as nn
from torch.nn import Linear

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
    def encoder_out(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        use_lip = self.config['model']['use_lipschitz']
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        x = torch.vstack((x0,x1))
        x = self.input_mapping(x)

        if use_lip:
            w = self.encoder[0].weight
            b = self.encoder[0].bias

            #c = self.encoder_lip[0].weight
            w = lip_norm(w)

            y = x@w.T+b
            x = torch.sin(2*y)

            for ii in range(1,self.nl+1):
                w = self.encoder[ii].weight
                b = self.encoder[ii].bias

                #c = self.encoder_lip[ii].weight
                w = lip_norm(w)

                y = x@w.T+b
                x  = torch.sin(2*y)

            # x = self.encoder[-1](x)
        else:
            x = torch.sin(self.encoder[0](x))

            for ii in range(1,self.nl):
                x = torch.sin(self.encoder[ii](x))

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

        scale0 = torch.sqrt(torch.sum (  ( x0**2 ) , dim =1)).unsqueeze(1) 
        x0 = x0/scale0

        scale1 = torch.sqrt(torch.sum (  ( x1**2 ) , dim =1)).unsqueeze(1) 
        x1 = x1/scale1

        x = x0*x1

        x = torch.sum(x,dim=1).unsqueeze(1)
        if model.config['model']['sym_div'] == True:
            sym_div_hype = model.config['model']['sym_div_hype']
            x = sym_div_hype * torch.acos(x-0.000001)/(scale0*scale1)
        else:
            sym_mult_hype = model.config['model']['sym_mult_hype']
            x = torch.acos(x-0.000001)*(scale0*scale1)/sym_mult_hype
        # x = 1000*torch.acos(x-0.000001)/(scale0*scale1) #250
        # x = torch.acos(x-0.000001)*(scale0*scale1)/50 #250
        return x, Xp
    
    @staticmethod
    def Loss(model, points, Yobs, beta):
        self = model
        tau, Xp = self.out(points)
        dtau = self.gradient(tau, Xp)
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        D_norm = torch.sqrt(torch.einsum('ij,ij->i', D, D))
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
        T00 = tau[:,0]**2

        Ypred0 = torch.sqrt(S0)
        Ypred1 = torch.sqrt(S1)
        sq_Yobs0 = Yobs[:,0]
        sq_Yobs1 = Yobs[:,1]
        l0 = sq_Yobs0*(Ypred0)
        l1 = sq_Yobs1*(Ypred1)
        loss0 = (torch.sqrt(l0)-1)**2
        loss1 = (torch.sqrt(l1)-1)**2
        diff = loss0 + loss1
        
        #
        # loss_n = torch.sum(diff*torch.exp(-0.1*tau[:, 0]))/Yobs.shape[0]
        # loss_n = torch.sum((loss0 + loss1)*torch.exp(-0.5*tau[:,0]))/Yobs.shape[0]
        # loss_n = torch.sum((loss0 + loss1)*torch.exp(-2.0*tau[:,0]))/Yobs.shape[0]
        loss_n_hype = model.config['model']['loss_n_hype']
        loss_n = torch.sum((loss0 + loss1)*torch.exp(loss_n_hype*tau[:,0]))/Yobs.shape[0] #-0.4
        # loss_n = torch.sum((loss0+loss1))


        
        loss = beta*loss_n #+ 1e-4*(reg_tau)

        return loss, loss_n, diff


    
    @staticmethod
    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        D_norm = torch.sqrt(torch.einsum('ij,ij->i', D, D))

        TT = tau[:,0]

        del Xp, tau, D, D_norm
        return TT
    
    @staticmethod
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.out(Xp)

        return tau


    @staticmethod
    def Speed(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))

        # tau
        tau, Xp = self.out(Xp)
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



