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
    def encoder_out1(submodel, coords):
        self = submodel
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        x = torch.vstack((x0, x1))
        x = self.input_mapping(x)

        w = self.encoder[0].weight
        b = self.encoder[0].bias
        #print(x.shape)
        y = x.unsqueeze(1)*w.unsqueeze(0)
        # y  = torch.sin(y)
        #print(y.shape)
        x = torch.sum(y,dim=2)
        x = torch.sin(self.encoder1[0](x))

        for ii in range(1,self.nl):
            w = self.encoder[ii].weight
            b = self.encoder[ii].bias
            y = x.unsqueeze(1)*w.unsqueeze(0)
            # y  = torch.sin(y)
            x = torch.sum(y,dim=2)#+b
            x = torch.sin(self.encoder1[ii](x))

        #'''
        w = self.encoder[-1].weight
        b = self.encoder[-1].bias
        #print(x.shape)
        y = x.unsqueeze(1)*w.unsqueeze(0)
        # y  = torch.sin(y)
        #print(y.shape)
        x = torch.sum(y,dim=2)#+b
        return x, coords
    
    @staticmethod
    def encoder_out(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        
        x = torch.vstack((x0,x1))

        #x = x.unsqueeze(1)
        
        x = self.input_mapping(x)

        #x  = torch.sin(self.encoder[0](x))
        #'''
        # w = self.encoder[0].weight
        # b = self.encoder[0].bias
        # #print(x.shape)
        # y = x.unsqueeze(1)*w.unsqueeze(0)
        # y  = torch.sin(y)
        # #print(y.shape)
        # x = torch.sum(y,dim=2)
        #'''
        #w = self.encoder1[ii].weight
        #y = x@w.T
        x = torch.sin(self.encoder[0](x))

        for ii in range(1,self.nl):
            #i0 = x
            #x = torch.sin(self.encoder[ii](x))
            #'''
            #w = self.encoder[ii].weight
            #b = self.encoder[ii].bias
            #print(x.shape)
            #y = x.unsqueeze(1)*w.unsqueeze(0)
            #y  = torch.sin(y)
            #print(y.shape)
            #x = torch.sum(y,dim=2)#+b
            #'''

            #w = self.encoder1[ii].weight
            #y = x@w.T
            #x = torch.sin(y)
            x = torch.sin(self.encoder1[ii](x))

        #'''
        #w = self.encoder[-1].weight
        #b = self.encoder[-1].bias
        #print(x.shape)
        #y = x.unsqueeze(1)*w.unsqueeze(0)
        #y  = torch.sin(y)
        #print(y.shape)
        #x = torch.sum(y,dim=2)#+b
        #'''
        x = self.encoder[-1](x)
        return x, coords
        #x = torch.sin(self.encoder[-1](x))
        #w = self.encoder[-1].weight
        #x = x@w.T
        #print(x.shape)
        x0 = torch.sin(x[:size,:])#.unsqueeze(1)
        x1 = torch.sin(x[size:,:])#.unsqueeze(1)
        #print(x0.shape)
        '''
        xx = torch.cat((x0-x1, x1-x0), dim=1)
        x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale
        x = x_0#torch.cat((x_0, x_1),1)*64
        '''
        x = (x0-x1)**2#/self.actout(x[:size,32]*x[size:,32]).unsqueeze(1)
        #x = (x0-x1)*torch.tanh((x0-x1))
        #print(x)
        #x = torch.exp(-0.01*torch.sum(x,dim=1).unsqueeze(1))/(math.e)

        w = torch.exp(10*self.autoscale[0].weight)

        x = torch.sum(x,dim=1).unsqueeze(1)#/(math.e)
        x = 10*torch.sqrt(x)

        
        
        return x, coords
    

    
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        
    @staticmethod
    def sym_op1(normalized_features, Xp):
        x0 = normalized_features[:Xp.shape[0], :]
        x1 = normalized_features[Xp.shape[0]:, :]
        x = (x0-x1)*torch.tanh((x0-x1))
        # x = (x0-x1)*(x0-x1)
        x = torch.sum(x, dim=1).unsqueeze(1)
        x = torch.sqrt(x+1e-6)
        return x, Xp
    
    @staticmethod
    def sym_op(normalized_features, Xp):
        x0 = torch.sin(normalized_features[:Xp.shape[0],:])#.unsqueeze(1)
        x1 = torch.sin(normalized_features[Xp.shape[0]:,:])#.unsqueeze(1)
        #print(x0.shape)
        '''
        xx = torch.cat((x0-x1, x1-x0), dim=1)
        x_0 = torch.logsumexp(self.scale*xx, 1)/self.scale
        x_1 = -torch.logsumexp(-self.scale*xx, 1)/self.scale
        x = x_0#torch.cat((x_0, x_1),1)*64
        '''
        x = (x0-x1)**2#/self.actout(x[:size,32]*x[size:,32]).unsqueeze(1)
        #x = (x0-x1)*torch.tanh((x0-x1))
        #print(x)
        #x = torch.exp(-0.01*torch.sum(x,dim=1).unsqueeze(1))/(math.e)

        # w = torch.exp(10*self.autoscale[0].weight)

        x = torch.sum(x,dim=1).unsqueeze(1)#/(math.e)
        x = 10*torch.sqrt(x)
        return x, Xp
    
    @staticmethod
    def Loss(self, points, Yobs, beta):
        
      
        tau, Xp = self.out(points)
        dtau = self.gradient(tau, Xp)
        
        #print(end-start)

        
        
        #tau, dtau, Xp = self.network.out_grad(points)
        
        #print(end-start)
        #print(dtau)

        #print(end-start)
        #print('')
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)#torch.norm(D, p=2, dim =1)**2
        
        
        DT0 = dtau[:,:self.dim]
        DT1 = dtau[:,self.dim:]

        T3    = tau[:,0]**2

        

        #print(tau.shape)

        

        #lap0 = 0.5*torch.einsum('ij,ij->i', DT0, DT0)#ltau[:,:self.dim].sum(-1) 
        #lap1 = 0.5*torch.einsum('ij,ij->i', DT1, DT1)#ltau[:,self.dim:].sum(-1) 
        
        
        
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)
       
        #0.001
        Ypred0 = torch.sqrt(S0)#torch.sqrt
        Ypred1 = torch.sqrt(S1)#torch.sqrt


        Ypred0_visco = Ypred0
        Ypred1_visco = Ypred1

        sq_Ypred0 = (Ypred0_visco)#+gamma*lap0
        sq_Ypred1 = (Ypred1_visco)#+gamma*lap1


        sq_Yobs0 = Yobs[:,0]#2*Yobs[:,0]*Yobs[:,0]*(1.5-Yobs[:,0])#**2
        sq_Yobs1 = Yobs[:,1]#2*Yobs[:,1]*Yobs[:,1]*(1.5-Yobs[:,1])#**2

        #loss0 = (sq_Yobs0/sq_Ypred0+sq_Ypred0/sq_Yobs0)#**2#+gamma*lap0
        #loss1 = (sq_Yobs1/sq_Ypred1+sq_Ypred1/sq_Yobs1)#**2#+gamma*lap1
        l0 = ((sq_Yobs0*(sq_Ypred0)))
        l1 = ((sq_Yobs1*(sq_Ypred1)))
        #loss0 = (l0+1/l0-2)#+gamma*lap0#**2
        #loss1 = (l1+1/l1-2)#+gamma*lap1#**2
        #loss0 = abs(l0-1)#**2#+gamma*lap0#**2
        #loss1 = abs(l1-1)#**2#+gamma*lap1#**2
        l0_2 = (torch.sqrt(l0))#**(1/4)
        l1_2 = (torch.sqrt(l1))#**(1/4)
        l0_4 = torch.sqrt(l0_2)#**(1/4)
        l1_4 = torch.sqrt(l1_2)#**(1/4)
        #loss0 = (l0+1/l0-2)#+gamma*lap0#**2
        #loss1 = (l1+1/l1-2)#+gamma*lap1#**2
        loss0 = (l0_2-1)**2
        loss1 = (l1_2-1)**2

        #loss0 = l0-4*l0_2*l0_4+6*l0_2-4*l0_4+1
        #loss1 = l1-4*l1_2*l1_4+6*l1_2-4*l1_4+1
        #diff = loss0 + loss1-4
        #loss_n = torch.sum((loss0 + loss1-4))/Yobs.shape[0]
        diff = loss0 + loss1 
        #weight = 0.000001/(Yobs[:,0]*Yobs[:,1]+1)
        #loss_n = torch.sum((loss0 + loss1 ))/Yobs.shape[0]

        weight = 0.001*(Yobs[:,0]*Yobs[:,1]+1)#min(0.0001,0.01*(epoch-200)/200)
        #
        loss_n = torch.sum(0.0001*(S0+S1)+(loss0 + loss1)*torch.exp(-0.5*tau[:,0]))/Yobs.shape[0] #time_diff

        
        loss = beta*loss_n #+ 1e-4*(reg_tau)

        return loss, loss_n, diff


    @staticmethod
    def Loss1(model, points, Yobs, beta):
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
        # loss0 = (l0-1)**2
        # loss1 = (l1-1)**2

        diff = loss0 + loss1
        loss_n = torch.sum(diff*torch.exp(-0.1*tau[:, 0]))/Yobs.shape[0]

        loss = beta*loss_n
        return loss, loss_n, diff
    
    @staticmethod
    def Loss2(self, points, Yobs, beta):
        tau, Xp = self.out(points)
        dtau = self.gradient(tau, Xp)
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


    @staticmethod
    def Tau1(model, Xp):
        self = model
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.out(Xp)

        return tau

    @staticmethod
    def Tau2(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.out(Xp)

        return tau

    @staticmethod
    def TravelTimes1(model, Xp):
        self = model
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = tau[:, 0]#torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT

    @staticmethod
    def TravelTimes2(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT
    
    @staticmethod
    def Speed1(model, Xp):
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

    @staticmethod
    def Speed2(self, Xp):
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

        T1    = 4*LogTau2*T0/T3*torch.einsum('ij,ij->i', DT1, DT1)
        T2    = 4*LogTau3/tau[:,0]*torch.einsum('ij,ij->i', DT1, D)

        
        
        S = (T1+T2+LogTau4)

        Ypred = 1/torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3, LogTau,LogTau2,LogTau3,LogTau4,DT1,S
        return Ypred


    @staticmethod
    def TravelTimes(self, Xp):
        # Apply projection from LatLong to UTM
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.out(Xp)

       
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = tau[:, 0]#torch.log(tau[:, 0])**2* torch.sqrt(T0)

        del Xp, tau, T0
        return TT
    
    @staticmethod
    def Tau(self, Xp):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.out(Xp)

        return tau

    @staticmethod
    def Speed(self, Xp):
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
    



