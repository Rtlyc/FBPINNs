import torch
import torch.nn as nn
from torch.nn import Linear

class SubNN(nn.Module):
    def __init__(self):
        super(SubNN, self).__init__()

    @staticmethod
    def init_network(nl, input_size, h_size):
        encoder = torch.nn.ModuleList()
        encoder1 = torch.nn.ModuleList()
        encoder.append(Linear(2*input_size, h_size))
        encoder1.append(Linear(h_size, h_size))

        for i in range(nl):
            encoder.append(Linear(h_size, h_size))
            encoder1.append(Linear(h_size, h_size))
        encoder.append(Linear(h_size, h_size))
        return encoder, encoder1
    
    @staticmethod
    def encoder_out(submodel, coords):
        self = submodel
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        x = torch.vstack((x0, x1))
        x = self.input_mapping(x)

        w = self.encoder[0].weight
        b = self.encoder[0].bias
        #print(x.shape)
        y = x.unsqueeze(1)*w.unsqueeze(0)
        y  = torch.sin(y)
        #print(y.shape)
        x = torch.sum(y,dim=2)
        x = torch.sin(self.encoder1[0](x))

        for ii in range(1,self.nl):
            w = self.encoder[ii].weight
            b = self.encoder[ii].bias
            y = x.unsqueeze(1)*w.unsqueeze(0)
            y  = torch.sin(y)
            x = torch.sum(y,dim=2)#+b
            x = torch.sin(self.encoder1[ii](x))

        #'''
        w = self.encoder[-1].weight
        b = self.encoder[-1].bias
        #print(x.shape)
        y = x.unsqueeze(1)*w.unsqueeze(0)
        y  = torch.sin(y)
        #print(y.shape)
        x = torch.sum(y,dim=2)#+b
        return x, coords
    

    
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
    @staticmethod
    def sym_op(normalized_features, Xp):
        x0 = normalized_features[:Xp.shape[0], :]
        x1 = normalized_features[Xp.shape[0]:, :]
        # x = (x0-x1)*torch.tanh((x0-x1))
        x = (x0-x1)*(x0-x1)
        x = torch.sum(x, dim=1).unsqueeze(1)
        x = torch.sqrt(x+1e-6)
        return x, Xp
    
    @staticmethod
    def Loss(model, points, Yobs, beta):
        self = model 
        tau, Xp = self.out(points)
        dtau = self.gradient(tau, Xp)

        DT0 = dtau[:, :self.dim]
        DT1 = dtau[:, self.dim:]
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
        sq_Ypred0 = torch.sqrt(S0)
        sq_Ypred1 = torch.sqrt(S1)
        sq_Yobs0 = Yobs[:, 0]
        sq_Yobs1 = Yobs[:, 1]

        l0 = sq_Yobs0*(sq_Ypred0)
        l1 = sq_Yobs1*(sq_Ypred1)
        sq_l0 = torch.sqrt(l0)
        sq_l1 = torch.sqrt(l1)

        loss0 = (sq_l0-1)**2 
        loss1 = (sq_l1-1)**2
        diff = loss0 + loss1
        loss_n = torch.sum(diff*torch.exp(-0.1*tau[:, 0]))/Yobs.shape[0]

        loss = beta*loss_n
        return loss, loss_n, diff
    

    @staticmethod
    def Tau(model, Xp):
        self = model
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.out(Xp)

        return tau


    @staticmethod
    def TravelTimes(model, Xp):
        self = model
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = tau[:, 0]#torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT

    @staticmethod
    def Speed(model, Xp):
        self = model 
        Xp = Xp.to(torch.device(self.Params['Device']))

        tau, Xp = self.out(Xp)
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

        
        S = torch.einsum('ij,ij->i', DT1, DT1)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3
        return Ypred
