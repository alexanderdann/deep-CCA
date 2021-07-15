from visuals import Visualizations as V
import torch
import time
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'legend.fontsize': 'x-large',
                     'text.usetex': True,
                 'figure.figsize': (10, 8),
                 'axes.labelsize': 'xx-large',
                 'axes.titlesize':'larger',
                 'xtick.labelsize':'xx-large',
                 'ytick.labelsize':'xx-large'})
plt.ion()
import seaborn as sns
import torch.nn as nn
import csv
from CCA import linear_cca
from TwoChannelModel import TwoChannelModel
import nn_init as nn_init
import numpy as np
from scipy.io import loadmat
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import MutualInformation as MI
sns.set()
sns.set_style('white')
sns.set_context('paper')


# Loading Spatialmaps
sP1 = loadmat('/Users/alexander/Documents/Uni/Bachelor/Thesis/simtb data/Version_JBSS_Proof1/set_subject_001_SIM.mat')['SM']
sP2 = loadmat('/Users/alexander/Documents/Uni/Bachelor/Thesis/simtb data/Version_JBSS_Proof1/set_subject_002_SIM.mat')['SM']


# Loading timecourse
TC1 = loadmat('/Users/alexander/Documents/Uni/Bachelor/Thesis/simtb data/Version_20x20_10000_5br/set_subject_001_SIM.mat')['TC']
TC2 = loadmat('/Users/alexander/Documents/Uni/bachelor/Thesis/simtb data/Version_20x20_10000_5br/set_subject_002_SIM.mat')['TC']



class DCCA():
    def __init__(self, epochs, input_nodes, hidden_layers, output_nodes, max_iter):
        self.reg_par = 1e-5
        self.learning_rate = 1e-3
        self.learning_rate_DAE = 0.5e-2
        self.device = torch.device('cpu')
        self.epochs = epochs
        self.linear_cca = linear_cca()
        self.nodes = [input_nodes] + hidden_layers + [output_nodes]

        self.nn_model = nn_init.NeuralNetwork(input_nodes, hidden_layers, output_nodes)
        self.optimizer = torch.optim.LBFGS(self.nn_model.parameters(), lr=self.learning_rate, max_iter=max_iter, line_search_fn=None)
        self.loss = self.nn_model.loss

    def fit(self, X_init, Y_init, S_x, S_y, outdim, rhos, path, counter):
        self.outdim = outdim

        X = torch.tensor(X_init)
        Y = torch.tensor(Y_init)

        X.to(self.device)
        Y.to(self.device)

        data_size = X.size(0)
        self.batch_size = Y.size(1)

        print(self.batch_size, data_size)

        train_losses = []

        last_out_X, last_out_Y = 0, 0

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.nn_model.train()
            def closure():
                self.optimizer.zero_grad()
                o1, o2 = self.nn_model(X, Y)
                last_out_X, last_out_Y = o1, o2
                loss = self.loss(o1, o2)
                print('loss:', loss.item())
                train_losses.append(int(loss.item()))
                loss.backward()
                return loss
            self.optimizer.step(closure)

            train_loss = np.mean(train_losses)
            epoch_time = time.time() - epoch_start_time
            print("\n\nTrainloss: " + str(train_loss))
            print('\nEpoch Number: ' + str(epoch))
            print("\n\nEpoch Time: " + str(epoch_time))



        _, outputs = self._get_outputs(X, Y)
        ccorValues, MIs_after, JBSS_res = \
            self.linear_cca.fit\
                (outputs[0], outputs[1], self.outdim, S_x, S_y, path, counter,  X_init, Y_init)

        return (ccorValues, MIs_after, JBSS_res)

    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)
            print("result: " + str(np.mean(losses)))
            #outputs = self.linear_cca.test(outputs[0], outputs[1])
            #print("test output: " + str(outputs))
            return np.mean(losses)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.nn_model.eval()
            losses = []
            out = []
            outputs1 = []
            outputs2 = []
            o1, o2 = self.nn_model(x1, x2)
            outputs1.append(o1)
            outputs2.append(o2)
            out.append(o1)
            out.append(o2)
            loss = self.loss(o1, o2)
            losses.append(loss.item())

        outputs = [out[0].cpu().numpy(),
                   out[1].cpu().numpy()]


        return losses, outputs


def mutualInfo(S_x, S_y):
    MIS_before = []
    for i in range(len(S_x.T)):
        MIS_before.append(MI.MI.mi_Kraskov([S_x.T[i], S_y.T[i]], 5))
        print(f'Mutual Information before: {MIS_before[i]}')
    return MIS_before

def testNN(samples, epochs, transform_mode, hidden_dim, output_dim, simulations, path):
    for M in samples:#[100, 500, 2500]:
        for iter in epochs:
            for transformation in transform_mode:#['def', 'sinh', 'ccubic', 'exp', 'softplus', 'ELU', 'tanh', 'sin', 'poly']:
                for dim_h in hidden_dim:#[50, 150, 300, 500, 1000]: #len(sP1[0])
                    for dim_o in output_dim:#[2, 8, 12, 24, len(sP1[0])]: #len(sP1)
                        for hidden_layers in range(len(dim_h)):

                            try:
                                os.makedirs(path)
                                print(f'Path: {path} exists: {os.path.exists(path)}\n\n')
                            except:
                                pass

                            # Defining emtpy lists
                            ccor, MIS_after, JBSS_colc = [], [], []

                            # num_signals, mod_sources and noise_list need to be the same size
                            num_signals = len(sP1)
                            noise_list = []
                            rhos = [0.85, 0.7, 0.3, 0]
                            mod_sources = [1, 2, 3, 4]

                            in_dim = len(sP1[0])
                            # hidden_dim = 0
                            # out_dim = num_signals
                            max_iter = 1
                            # degree for polynomial transformation
                            degree = 4
                            # rhos = [0.95, 0.7, 0.0]
                            # rhos = [0.8, 0.7, 0.6, 0.00 , 0.00]
                            # mixings: def, sin, exp, reci, sqr, descsin
                            mix = transform_mode

                            # mod_sources = [1, 2, 3]
                            use_same_signal = True

                            # SOURCE = TwoChannelModel(use_same_signal, sP1, sP2, TC1, TC2, rhos, num_signals, observations).TCM(part_mix=mod_sources, mixing=mix, amp=0.5, stretch=180, noise_var_x=0.1, noise_var_y=0.1)

                            # initial()

                            SOURCE = TwoChannelModel(use_same_signal, sP1, sP2, TC1, TC2, rhos, path, num_signals, M).\
                                TCM(part_mix=mod_sources, mixing=mix, amp=1.5, stretch=120, noise_var_x=noise_list,
                                        noise_var_y=noise_list, degree=degree)

                            cre_rhos = SOURCE[4]

                            print(f'Created rhos: {cre_rhos}')
                            print(SOURCE[0].T[0].shape)

                            MIS_before = mutualInfo(SOURCE[2], SOURCE[3])

                            for counter in range(simulations):
                                try:
                                    dcca_model = DCCA(iter, in_dim, hidden_layers, out_dim, max_iter)
                                    dcca_result, MI_after, JBSS_res = dcca_model.fit(SOURCE, out_dim, rhos, path,
                                                                                     counter)
                                    ccor.append(dcca_result)
                                    MIS_after.append(np.array(MI_after))
                                    JBSS_colc.append(JBSS_res)
                                    # dcca_model.test(torch.tensor(SOURCE[0]), torch.tensor(SOURCE[1]))
                                except Exception as excp1:
                                    print(f'ERROR: {excp1}')
                            try:
                                visuals.visual_MI(MIS_before, MIS_after, 'Kraskov', path)
                                visuals.visual_JBSS(JBSS_colc, path)
                                visuals.visual_Corr(ccor, cre_rhos, path)
                            except Exception as excp2:
                                print(f'ERROR: {excp2}')

#visuals.visual_MI(MIs_before, MIs_after, 'Kraskov', path)
#visuals.visual_JBSS(JBSS_collection, path)
#visuals.visual_Corr(canonicalCorr, createdRhos, path)

epochs = 250
in_dim = 64
hidden_layers = [64]
out_dim = 4
max_iter = 1

rhos = [0.7, 0.6, 0.0, 0.0]
observations = 500
dim_data = 64
degree = 1
mod_sources = [0, 1, 2, 3]
transformation1 = 'exp'
transformation2 = 'sinh'

keys = time.asctime(time.localtime(time.time())).split()

path = '/Users/alexander/Documents/Uni/Work/deepCCA/Simulation/' + str('-'.join(keys))

try:
    os.makedirs(path)
    print(f'Path: {path} exists: {os.path.exists(path)}\n\n')
except:
    pass

X, Y, S_x, S_y, created_rhos = TwoChannelModel(sP1, sP2, path, rhos, observations, dim_data).\
    transform(part_mix=mod_sources, mixing1=transformation1, mixing2=transformation2, amp=1.5, stretch=120, degree=degree)

print(f'Created rhos: {created_rhos}')

cCorr_Collection = []
MIs_Collection = []
JBSS_Collection = []

for counter in range(20):
    dcca_model = DCCA(epochs, in_dim, hidden_layers, out_dim, max_iter)
    dcca_result, MIs_after, JBSS_res = dcca_model.fit(X, Y, S_x, S_y, out_dim, rhos, path, counter)
    cCorr_Collection.append(dcca_result)
    MIs_Collection.append(np.array(MIs_after))
    JBSS_Collection.append(JBSS_res)

#counter = 0
#dcca_result, MIs_after, JBSS_res = linear_cca().fit(X, Y, out_dim, S_x, S_y, path, counter, X, Y)
#cCorr_Collection.append(dcca_result)
#MIs_Collection.append(np.array(MIs_after))
#JBSS_Collection.append(JBSS_res)

V().visual_JBSS(JBSS_Collection, path)
V().visual_Corr(cCorr_Collection, created_rhos, path)

