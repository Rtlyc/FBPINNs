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
    def encoder_out(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))
        
        x = self.input_mapping(x)
        x = torch.sin(self.encoder[0](x))

        for ii in range(1,self.nl):
            x = torch.sin(self.encoder1[ii](x))

        x = self.encoder[-1](x)
        return x, coords

    

    
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
    
    @staticmethod
    def sym_op(normalized_features, Xp):
        x0 = normalized_features[:Xp.shape[0],:]#.unsqueeze(1)
        x1 = normalized_features[Xp.shape[0]:,:]#.unsqueeze(1)
        #print(x0.shape)
        '''
        xx = torch.cat((x0-x1, x1-x0), dim=1)
        x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale
        x = x_0#torch.cat((x_0, x_1),1)*64
        '''
        x = (x0-x1)**2#/self.actout(x[:size,32]*x[size:,32]).unsqueeze(1)
        # x = (x0-x1)*torch.tanh((x0-x1))
        #print(x)
        #x = torch.exp(-0.01*torch.sum(x,dim=1).unsqueeze(1))/(math.e)

        # w = torch.exp(10*self.autoscale[0].weight)

        x = torch.sum(x,dim=1).unsqueeze(1)#/(math.e)
        x = 1*torch.sqrt(x)
        return x, Xp
    
    @staticmethod
    def Loss(self, points, Yobs, beta):
        tau, Xp = self.out(points)
        dtau = self.gradient(tau, Xp)
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        D_norm = torch.sqrt(torch.einsum('ij,ij->i', D, D))
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
        T00 = tau[:,0]**2
        T10 = torch.einsum('ij,ij->i', D, D)*S0 
        T11 = torch.einsum('ij,ij->i', D, D)*S1
        T20 = 2*torch.einsum('ij,ij->i', D, DT0)*tau[:,0]
        T21 = 2*torch.einsum('ij,ij->i', D, DT1)*tau[:,0]
        above0 = T00 + T10 - T20
        above1 = T00 + T11 - T21
        under = tau[:,0]**4
        
        Ypred0 = torch.sqrt(above0/under)
        Ypred1 = torch.sqrt(above1/under)
        sq_Yobs0 = Yobs[:,0]
        sq_Yobs1 = Yobs[:,1]
        l0 = sq_Yobs0*(Ypred0)
        l1 = sq_Yobs1*(Ypred1)
        loss0 = (torch.sqrt(l0)-1)**2
        loss1 = (torch.sqrt(l1)-1)**2
        diff = loss0 + loss1
        
        #
        # loss_n = torch.sum(diff*torch.exp(-0.1*tau[:, 0]))/Yobs.shape[0]
        loss_n = torch.sum((loss0 + loss1))/Yobs.shape[0]

        
        loss = beta*loss_n #+ 1e-4*(reg_tau)

        return loss, loss_n, diff


    
    @staticmethod
    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        TT = torch.sqrt(torch.einsum('ij,ij->i', D, D))/tau[:,0]

        del Xp, tau, D
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
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        # D_norm = torch.sqrt(torch.einsum('ij,ij->i', D, D))

        DT1 = dtau[:,self.dim:]
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
        T00 = tau[:,0]**2
        T11 = torch.einsum('ij,ij->i', D, D)*S1
        T21 = 2*torch.einsum('ij,ij->i', D, DT1)*tau[:,0]
        above1 = T00 + T11 - T21
        under = tau[:,0]**4
        
        Ypred = 1/torch.sqrt(above1/under)

        
        del Xp, tau, dtau, T00, T11, T21, above1, under
        return Ypred



