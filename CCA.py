import random

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import scipy
import math
import os
import csv
# from sklearn import preprocessing
from MutualInformation import MutualInfo
import time
import scipy.linalg as sl
from numpy.linalg import multi_dot, inv, svd, matrix_power
from scipy.linalg import sqrtm
from scipy.io import loadmat
from numpy.random import default_rng

rand = default_rng()
from matplotlib.colors import ListedColormap

sns.set()
sns.set_style('white')
sns.set_context('paper')


class linear_cca():
    def __init__(self):
        pass

    def mutualInfo(self, outdim_size, epsi, omeg):
        MIS_after = []
        for i in range(outdim_size):
            if outdim_size == 1:
                MIS_after.append(MutualInfo.mi_Kraskov([epsi[i].T, omeg[i].T], 5))
            else:
                MIS_after.append(MutualInfo.mi_Kraskov([epsi[i], omeg[i]], 5))
            print(f'Mutual Information after: {MIS_after[i]}')

        return MIS_after

    def _get_result(self, x, idx):
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = np.dot(result, self.w[idx])
        return result

    def test(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)

    def jbss(self, est_source1, est_source2, real_source1, real_source2, outdim, H1, H2, counter):
        M = len(real_source1)
        print(f'Shape of est_source {est_source1[0:outdim].shape}')
        print(f'Shape of real_source {real_source1.shape}')

        # Correlation coefficients
        cor_real_est_X = np.zeros((len(est_source1[0:outdim]), len(real_source1[0])))
        cor_real_est_Y = np.zeros((len(est_source2[0:outdim]), len(real_source2[0])))

        print(cor_real_est_X.shape)

        dim1 = -1
        for est1 in est_source1[0:outdim].real:
            dim1 += 1
            dim2 = 0
            for real1 in real_source1.T:
                stddev_est = np.sqrt(np.sum([x ** 2 for x in est1]) / (len(est1) - 1))
                stddev_real = np.sqrt(np.sum([x ** 2 for x in real1]) / (len(real1) - 1))
                coef = np.dot(est1, real1) / (stddev_est * stddev_real * len(est1))
                # print(f'Coefficient 1: {coef}')
                cor_real_est_X[dim1][dim2] = abs(coef)
                dim2 += 1

        dim1 = -1
        for est1 in est_source2[0:outdim].real:
            dim1 += 1
            dim2 = 0
            for real1 in real_source2.T:
                stddev_est = np.sqrt(np.sum([x ** 2 for x in est1]) / (len(est1) - 1))
                stddev_real = np.sqrt(np.sum([x ** 2 for x in real1]) / (len(real1) - 1))
                coef = np.dot(est1, real1) / (stddev_est * stddev_real * len(est1))
                # print(f'Coefficient 2: {coef}')
                cor_real_est_Y[dim1][dim2] = abs(coef)
                dim2 += 1

        dim_eps = len(est_source2[0:outdim].real)
        dim_rl = len(real_source2.T)

        print(f'Estimated Covariance X: {cor_real_est_X}')
        print(f'Estimated Covariance Y: {cor_real_est_Y}')
        print(f'Shape of JBSS Covariance Test: {cor_real_est_Y.shape}')

        sns.set_context('paper')
        sns.set_style('white')
        plt.rcParams.update({'figure.figsize': (int(dim_rl), int(dim_eps + 1))})
        colors = sns.color_palette(['#E0E0E0', '#00ff1f', '#FFFF33', '#ffa500', '#ff0000', ]).as_hex()
        cmap = ListedColormap(colors)
        legend = plt.matshow(cor_real_est_X, cmap='Blues')
        clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.03)
        for t in clrbr.ax.get_xticklabels():
            t.set_fontsize(20)
        plt.clim(0, 1)
        clrbr.set_label(r'Correlation', fontsize=18)
        plt.ylabel(r'$\hat{\mathbf{\varepsilon}}$', fontsize=22)
        # plt.suptitle('Correlations between True and Estimated Sources', fontweight='bold', fontsize=26)
        plt.title(r'$\mathbf{S}_{\mathrm{X}}$', fontsize=20)
        plt.yticks(fontsize=22)
        plt.xticks(fontsize=22)
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False)
        plt.tight_layout()
        full_path = self.path + '/' + str(counter) + '/' + 'CTE1.png'
        plt.savefig(full_path)
        plt.show(block=False)
        print(cor_real_est_X)

        legend = plt.matshow(cor_real_est_Y, cmap='Blues')
        clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.03)
        for t in clrbr.ax.get_xticklabels():
            t.set_fontsize(20)
        plt.clim(0, 1)
        clrbr.set_label(r'Correlation', fontsize=18)
        # plt.suptitle('Correlations between True and Estimated Sources', fontweight='bold', fontsize=26)
        plt.ylabel(r'$\hat{\mathbf{\omega}}$', fontsize=22)
        plt.yticks(fontsize=22)
        plt.xticks(fontsize=22)
        plt.title(r'$\mathbf{S}_{\mathrm{Y}}$', fontsize=20)
        full_path = self.path + '/' + str(counter) + '/' + 'CTE2.png'
        plt.tight_layout()
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False)
        plt.savefig(full_path)
        plt.show(block=False)
        print(cor_real_est_Y)
        plt.close('all')

        print(f'Shape of corr_real_est X {cor_real_est_X.shape}')
        print(f'Shape of corr_real_est Y {cor_real_est_Y.shape}')

        return (cor_real_est_X, cor_real_est_Y)

    def fit(self, H1, H2, outdim_size, source_H1, source_H2, path, counter, initial_X, initial_Y):
        self.path = path
        os.mkdir(self.path + '/' + str(counter))


        def sqrtm_inv(val):
            tmp = sqrtm(inv(val))
            return tmp

        r1 = 1e-2
        r2 = 1e-2

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        meanH1 = np.mean(H1, axis=0)
        meanH2 = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(meanH1, (m, 1))
        H2bar = H2 - np.tile(meanH2, (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)

        SigmaHat11_root_inv = sqrtm_inv(SigmaHat11)
        SigmaHat22_root_inv = sqrtm_inv(SigmaHat22)
        SigmaHat22_root_inv_T = SigmaHat22_root_inv.T

        C = multi_dot([SigmaHat11_root_inv, SigmaHat12, SigmaHat22_root_inv_T])
        U, D, V = svd(C)

        A = np.dot(U.T, SigmaHat11_root_inv)
        B = np.dot(V, SigmaHat22_root_inv)

        epsi = np.dot(A, H1.T)
        omeg = np.dot(B, H2.T)

        print('Shape')
        print(epsi.shape)

        est_X = epsi[:, 0:outdim_size]
        est_Y = omeg[:, 0:outdim_size]

        plt.rcParams.update({'figure.figsize': (5, 4)})
        plt.xlabel(r'$\varepsilon$', fontweight='bold', fontsize='22')
        plt.ylabel(r'$\omega$', fontweight='bold', fontsize='22')
        plt.scatter(epsi[0], omeg[0], c='r', marker='.')
        plt.tight_layout()
        full_path = self.path + '/' + str(counter) + '/' + 'INTREP.png'
        plt.savefig(full_path)
        plt.show()

        JBSS_res = self.jbss(epsi, omeg, np.array(source_H1), np.array(source_H2), outdim_size, np.array(initial_X),
                             np.array(initial_Y), counter)

        print("Canonical Correlations: " + str(D[0:outdim_size]))

        MIs_after = self.mutualInfo(outdim_size, epsi, omeg)

        return D[0:outdim_size], MIs_after, JBSS_res

