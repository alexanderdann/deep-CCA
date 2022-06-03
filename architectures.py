import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sqrtm import sqrtm
import numpy as np
from utilities import EarlyStoppingCallback, visualize_brain_maps

torch.set_default_tensor_type(torch.DoubleTensor)


def CCA(view1, view2, r1=0, r2=0, shared_dim=None):
    V1, V2 = torch.from_numpy(view1), torch.from_numpy(view1)
    o1 = o2 = V1.size(0)
    m = V1.size(1)
    
    assert V1.size(1) == V2.size(1)
    if shared_dim is None:
        shared_dim = np.amin([V1.size(0), V2.size(0)])
    
    V1_bar = V1 - V1.mean(dim=1).unsqueeze(dim=1)
    V2_bar = V2 - V2.mean(dim=1).unsqueeze(dim=1)

    
    SigmaHat12 = torch.matmul(V1_bar, V2_bar.t()) / (m - 1)
    SigmaHat11 = torch.matmul(V1_bar, V1_bar.t()) / (m - 1) + r1 * torch.eye(o1)
    SigmaHat22 = torch.matmul(V2_bar, V2_bar.t()) / (m - 1) + r2 * torch.eye(o2)
    
    SigmaHat11RootInv = sqrtm(torch.inverse(SigmaHat11))
    SigmaHat22RootInv = sqrtm(torch.inverse(SigmaHat22))

    C = torch.linalg.multi_dot([SigmaHat11RootInv, SigmaHat12, SigmaHat22RootInv])
    
    U, D, V = torch.linalg.svd(C, full_matrices=False)
    
    A = torch.matmul(U.t()[:shared_dim], SigmaHat11RootInv)
    B = torch.matmul(V.t()[:shared_dim], SigmaHat22RootInv)

    epsilon = torch.matmul(A, V1_bar)
    omega = torch.matmul(B, V2_bar)
    
    assert (A.size(0) == B.size(0)) and (A.size(0) == shared_dim), print('Shared dimension bigger then dimension of data.')
    return A, B, epsilon, omega, D


def _connect_neurons(input_nodes, hidden_layers, output_nodes):
        layers = list()
        layers.extend([nn.Linear(input_nodes, hidden_layers[0]),
                          nn.Sigmoid(),
                          nn.BatchNorm1d(num_features=hidden_layers[0], affine=False)])
        
        for layer_idx, layer in enumerate(hidden_layers[:-1]):
            layers.extend([nn.Linear(layer, hidden_layers[layer_idx+1]),
                           nn.Sigmoid(),
                           nn.BatchNorm1d(num_features=hidden_layers[layer_idx+1], affine=False)])
            
        layers.extend([nn.Linear(hidden_layers[-1], output_nodes)])
        
        return nn.Sequential(*layers)


class deepCCA(nn.Module):
    def __init__(self, input_nodes, hidden_layers, output_nodes):
        super(deepCCA, self).__init__()
        
        self.loss = loss_fn()# LossFunction().loss# loss_fn()
        self.model1 = _connect_neurons(input_nodes[0], hidden_layers[0], output_nodes)
        self.model2 = _connect_neurons(input_nodes[1], hidden_layers[1], output_nodes)
        
        print(f'Model for the 1st view\n\n{self.model1}\n\nModel for the 2nd view\n\n{self.model2}')
        
        
    def forward(self, view1, view2):
        if (type(view1) == type(view1) == np.ndarray):
            view1, view2 = torch.from_numpy(view1).t(), torch.from_numpy(view2).t()

        return self.model1(view1), self.model2(view2)
    
    
    def fit(self, view1, view2, l2_param, learning_rate, epochs, early_stopping_idx, early_stopping_epsilon=1e-4, LOGPATH='LOG', optim='Adam'):
        view1, view2 = torch.from_numpy(view1).t(), torch.from_numpy(view2).t()
        view1.to(torch.device('cpu'))
        view2.to(torch.device('cpu'))
        losses = list()
        
        if optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            
        elif optim == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=learning_rate)
        
        else:
            raise ValurError

        writer = SummaryWriter(LOGPATH)
        
        for epoch_idx in tqdm(range(epochs)):
            if EarlyStoppingCallback().check_last_epochs(losses, early_stopping_idx, early_stopping_epsilon):
                break
            
            if optim == 'Adam':
                optimizer.zero_grad()
                fy_1, fy_2 = self(view1, view2)
                cca_loss = self.loss(fy_1, fy_2)
                reg_loss = l2_param * torch.sum(torch.tensor([torch.sum(torch.pow(param, 2)) for param in self.parameters()]))
                loss = torch.add(cca_loss, reg_loss)
                loss.backward()
                losses.append(loss)

                if epoch_idx % 10 == 0:
                    S_X, S_Y = visualize_brain_maps(fy_1.detach().t().numpy(), fy_2.detach().t().numpy(), tensorboard=True)
                    writer.add_image(f'{LOGPATH}/S_X_hat', S_X, epoch_idx, dataformats='HW')
                    writer.add_image(f'{LOGPATH}/S_Y_hat', S_Y, epoch_idx, dataformats='HW')
                    writer.add_graph(self, [view1, view2])
                    writer.close()

                if epoch_idx % 5 == 0:
                    writer.add_scalar('Losses/CCA Loss', cca_loss, epoch_idx)
                    writer.add_scalar('Losses/L2 Loss', reg_loss, epoch_idx)
                    writer.add_scalar('Losses/Total Loss', loss, epoch_idx)
                    writer.close()

                optimizer.step()  
                
            elif optim == 'LBFGS':
                
                def closure():
                    optimizer.zero_grad()
                    fy_1, fy_2 = self(view1, view2)
                    cca_loss = self.loss(fy_1, fy_2)
                    reg_loss = l2_param * torch.sum(torch.tensor([torch.sum(torch.pow(param, 2)) for param in self.parameters()]))
                    loss = torch.add(cca_loss, reg_loss)
                    loss.backward()
                    losses.append(loss)

                    if epoch_idx % 10 == 0:
                        S_X, S_Y = visualize_brain_maps(fy_1.detach().t().numpy(), fy_2.detach().t().numpy(), tensorboard=True)
                        writer.add_image(f'{LOGPATH}/S_X_hat', S_X, epoch_idx, dataformats='HW')
                        writer.add_image(f'{LOGPATH}/S_Y_hat', S_Y, epoch_idx, dataformats='HW')
                        writer.add_graph(self, [view1, view2])
                        writer.close()

                    if epoch_idx % 5 == 0:
                        writer.add_scalar('Losses/CCA Loss', cca_loss, epoch_idx)
                        writer.add_scalar('Losses/L2 Loss', reg_loss, epoch_idx)
                        writer.add_scalar('Losses/Total Loss', loss, epoch_idx)
                        writer.close()

                    return loss
            
                optimizer.step(closure)  

        
class loss_fn(nn.Module):
    def __init__(self):
        super(loss_fn, self).__init__()
        
    def __call__(self, H1, H2, r1 = 2e-2, r2 = 2e-2, eps = 1e-9):
        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)
        m = H1.size(1)
        
        H1bar = torch.subtract(H1, H1.mean(dim=1).unsqueeze(dim=1))
        H2bar = torch.subtract(H2, H2.mean(dim=1).unsqueeze(dim=1))

        SigmaHat12 = torch.matmul(H1bar, H2bar.t()) / (m - 1)
        SigmaHat11 = torch.add(torch.matmul(H1bar, H1bar.t()) / (m - 1), r1 * torch.eye(o1))
        SigmaHat22 = torch.add(torch.matmul(H2bar, H2bar.t()) / (m - 1), r2 * torch.eye(o2))
        
        [D1, V1] = torch.linalg.eigh(SigmaHat11, UPLO='U')
        [D2, V2] = torch.linalg.eigh(SigmaHat22, UPLO='U')
        
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        
        SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())
        
        Tval = torch.linalg.multi_dot([SigmaHat11RootInv, SigmaHat12, SigmaHat22RootInv])
        TT = torch.matmul(Tval.t(), Tval)
        
        return -torch.trace(sqrtm(TT))

