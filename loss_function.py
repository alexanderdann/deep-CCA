import torch
import numpy as np
from sqrtm import sqrtm
#from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


class LossFunction:
    def __init__(self):
        self.device = torch.device('cpu')
        self.outdim_size = 6
        self.use_all_singular_values = True

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """
        r1 = 2e-2
        r2 = 2e-2
        lambda_reg = 2e-3
        eps = 1e-9
        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = o2 = H1.size(0)

        m = H1.size(1)
        #         print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)
            # assert torch.isnan(SigmaHat11).sum().item() == 0
            # assert torch.isnan(SigmaHat12).sum().item() == 0
            # assert torch.isnan(SigmaHat22).sum().item() == 0

        def is_pos_def(x):
            eig, vec = torch.eig(x, False)
            return eig

        for x in [SigmaHat11, SigmaHat22, SigmaHat12]:
            print(f'Is positive semidefinit: {is_pos_def(x)}')
            # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
            # assert torch.isnan(D1).sum().item() == 0
            # assert torch.isnan(D2).sum().item() == 0
            # assert torch.isnan(V1).sum().item() == 0
            # assert torch.isnan(V2).sum().item() == 0

            # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
            # print(posInd1.size())
            # print(posInd2.size())

        SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.chain_matmul(SigmaHat11RootInv, SigmaHat12, SigmaHat22RootInv)
            #         print(Tval.size())


        #how to work with singular values
        if self.use_all_singular_values:
                # all singular values are used to calculate the correlation
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.ones(trace_TT.shape[0]) * lambda_reg).to(self.device))  # regularization for more stability
            corr = torch.trace(sqrtm(trace_TT))
                # assert torch.isnan(corr).item() == 0
        else:
                # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0]) * lambda_reg).to(self.device))  # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U > eps, U, (torch.ones(U.shape).double() * eps).to(self.device))
            #U = U.topk(self.outdim_size)[0]
            print(U.detach().numpy())
            corr = torch.sum(sqrtm(U.detach().numpy()))

        return -corr

    def calculate(self, H1, H2):
        r_1 = 1e-4
        r_2 = 1e-4
        eps = 1e-9

        H_1, H_2 = H1, H2
        m = H_1.shape[1]
        o1 = H_1.shape[0]
        o2 = H_2.shape[0]


        diag = torch.eye(m, device=self.device)

        H_1_bar = H_1 - (1 / m) * torch.matmul(H_1, diag)
        H_2_bar = H_2 - (1 / m) * torch.matmul(H_2, diag)

        sigma_12 = (1/(m - 1)) * torch.matmul(H_1_bar, H_2_bar.t())
        sigma_11 = (1/(m - 1)) * torch.matmul(H_1_bar, H_1_bar.t()) + r_1 * torch.eye(o1, device=self.device)
        sigma_22 = (1/(m - 1)) * torch.matmul(H_2_bar, H_2_bar.t()) + r_2 * torch.eye(o2, device=self.device)

        [D1, V1] = torch.symeig(sigma_11, eigenvectors=True)
        [D2, V2] = torch.symeig(sigma_22, eigenvectors=True)

        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        sigma_11_minus_half = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        sigma_22_minus_half = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        T = torch.chain_matmul(sigma_11_minus_half, sigma_12, sigma_22_minus_half)
        T_T = T.t()

        trace_TT = torch.matmul(T_T, T)
        trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0]) * r_1).to(self.device))  # regularization for more stability
        corr_H_1_H_2 = torch.trace(sqrtm(trace_TT))

        #tmp = torch.sqrt(torch.matmul(T_T, T))

        # sqrt?????
        #corr_H_1_H_2 = torch.trace(tmp)

        return -corr_H_1_H_2

        #U, D, V_T = torch.svd(T)

        #D = torch.diag(D)
        #U_T = U.t()

        #nabla_11 = -0.5 * torch.chain_matmul(sigma_11_minus_half, U, D, U_T, sigma_11_minus_half)
        #nabla_12 = torch.chain_matmul(sigma_11_minus_half, U, V_T, sigma_22_minus_half)
        #nabla_21 = 0
        #nabla_22 = 0

        #corr_H1 = (1 / m - 1) * ((2 * nabla_11 * H_mod_1) + (nabla_12 * H_mod_2))
        #corr_H2 = 0