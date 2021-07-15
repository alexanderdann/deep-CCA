import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as PathEffects
plt.rcParams.update({'legend.fontsize': 'x-large',
                     'text.usetex': True,
                 'figure.figsize': (10, 8),
                 'axes.labelsize': 'xx-large',
                 'axes.titlesize':'larger',
                 'xtick.labelsize':'xx-large',
                 'ytick.labelsize':'xx-large'})

sns.set()
sns.set_style('white')
sns.set_context('paper')

class Visualizations:
    def __init__(self):
        pass

    def visual_Corr(self, ccor_list, rhos, path):
        ccor = self.avg(ccor_list)[0:len(rhos)]

        plt.rcParams.update({'figure.figsize': (6, 6)})
        sns.set_style('white')
        sns.set_context('notebook')
        fig, ax = plt.subplots()
        #plt.suptitle(r'Canonical Correlations vs. True Correlations', fontweight='bold')
        plt.xlabel(r'Number of Component', fontweight='bold', fontsize='22')
        plt.ylabel(r'Correlation', fontweight='bold', fontsize='22')
        x_l = np.linspace(1, len(rhos), len(rhos))
        plt.xticks(x_l, fontsize=24)
        legend = plt.scatter(x_l, rhos, c='black', marker='o', s=300)
        plt.yticks(np.linspace(0, 1, 6), fontsize=24)
        legend.set_label(r'True Correlations')
        print(f'ccor {ccor} x {rhos}')
        legend = plt.scatter(x_l, ccor, c='deepskyblue', marker='x', s=300)
        error = np.array(rhos) - np.array(ccor)
        plt.plot([0.5, len(rhos)+0.5], [-0.1, -0.1], color='black', linewidth=1)
        ax.annotate(r'$\left | \mathrm{err} \right |$', (0.8, -0.16), fontsize=18, va='center', ha='center', fontweight='bold')
        for ind in range(len(error)):
            ax.annotate(f'{np.abs(np.around(error[ind], 2))}', (ind+1, -0.25), fontsize=18, va='center', ha='center', color='red')

        legend.set_label(r'Estimated Canonical Correlations')
        plt.xlim(0.5, len(rhos)+0.5)
        plt.ylim(-0.3, 1.05)
        #leg = plt.legend(bbox_to_anchor=(1, 1.05), loc=2, borderaxespad=1.,prop={'size': 15})
        leg = plt.legend( loc='upper right', borderaxespad=-4. ,prop={'size': 15})
        leg.get_frame().set_facecolor('lightgray')
        plt.tight_layout()
        full_path = path + '/' + 'CCOR.png'
        plt.savefig(full_path)
        plt.show(block=False)
        plt.close('all')

    def avg(self, elems):
        avg_elems = []
        for i in range(len(elems[0])):
            tmp = []
            for j in range(len(elems)):
                tmp.append(elems[j][i])
            avg_elems.append(np.mean(tmp))

        return avg_elems

    def visual_MI(self, MIS_bef, MIS_after, name, path):
        MIS_before = np.sort(np.array(MIS_bef))
        MIS_after = np.sort(np.array(MIS_after))

        MIS_before = MIS_before[::-1].copy()
        #MIS_after = MIS_after[::-1].copy()

        print(f'MIS before {MIS_before}\n')

        avg_MIS_after = np.mean(MIS_after, axis=0)[::-1][0:4]
        plt.rcParams.update({'figure.figsize': (6, 6)})
        sns.set_style('white')
        sns.set_context('notebook')
        if name == 'Kraskov':
            full_path = path + '/' + 'KraskovMI.png'
        else:
            full_path = path + '/' + 'MI.png'

        fig, ax = plt.subplots()
        plt.xlabel('Number of Component', fontweight='bold', fontsize='22')
        plt.ylabel('Mutual Information', fontweight='bold', fontsize='22')
        x = np.linspace(1, len(MIS_before), len(MIS_before))
        plt.xticks(x, fontsize=24)
        plt.yticks(np.linspace(0, 1, 6), fontsize=24)
        error = np.array(MIS_before) - np.array(avg_MIS_after)
        check = np.concatenate((avg_MIS_after, MIS_before))
        print(f'check {check}')
        plt.plot([0.5, len(avg_MIS_after)+0.5], [-0.1, -0.1], color='black', linewidth=1)

        legend = ax.scatter(x, MIS_before, c='black', marker='o', s=300)
        legend.set_label('Mutual Information before Deep CCA')
        plt.xlim(0.5, len(MIS_before) + 0.5)
        plt.ylim(-0.3, 1.05)
        ax.annotate(r'$\left | \mathrm{err} \right |$', (0.8, -0.16), fontsize=18, va='center', ha='center',
                    fontweight='bold')
        for ind in range(len(error)):
            ax.annotate(f'{np.abs(np.around(error[ind], 2))}', (ind + 1, -0.25), fontsize=18, va='center', ha='center',
                        color='red')
        legend = ax.scatter(x, avg_MIS_after[0:4], c='deepskyblue', marker='x', s=300)
        legend.set_label('Mutual Information after Deep CCA')
        leg = plt.legend(loc='upper right', borderaxespad=-4., prop={'size': 15})
        leg.get_frame().set_facecolor('lightgray')
        plt.tight_layout()
        #plt.legend()
        plt.savefig(full_path)
        plt.show(block=False)
        plt.close('all')

    def visual_JBSS(self, collection, path):
        JBSS_x, JBSS_y = [], []
        for element in collection:
            JBSS_x.append(np.array(element[0]))
            JBSS_y.append(np.array(element[1]))

        avg_JBSS_x = np.mean(JBSS_x, axis=0)
        avg_JBSS_y = np.mean(JBSS_y, axis=0)


        sns.set_context('paper')
        sns.set_style('white')
        # Adjust for JBSS Plots for needed dimensions
        if len(avg_JBSS_x) == 4:
            plt.rcParams.update({'figure.figsize': (int(len(avg_JBSS_x)+2), int(len(avg_JBSS_x[0])+2))})
        else:
            pass
            #plt.rcParams.update({'figure.figsize': (int(len(avg_JBSS_x)*2), int(len(avg_JBSS_x)*2))})

        colors = sns.color_palette(['#E0E0E0', '#00ff1f', '#FFFF33', '#ffa500', '#ff0000', ]).as_hex()
        cmap = ListedColormap(colors)

        font = {'style': 'italic',
                'weight': 'extra bold',
                'size': 20}

        mpl.rc('font', **font)

        print(f'\nShape of JBSS x avg: {avg_JBSS_x.shape}\n')
        print(f'Shape of JBSS y avg: {avg_JBSS_y.shape}\n')


        fig, ax = plt.subplots()
        legend = ax.matshow(avg_JBSS_x, cmap='Blues')
        clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.03)
        for t in clrbr.ax.get_xticklabels():
            t.set_fontsize(20)
        legend.set_clim(0, 1)
        clrbr.set_label(r'Correlation', fontsize=18)
        plt.ylabel(r'$\hat{\mathbf{\varepsilon}}$', fontsize=40)
        #plt.suptitle(r'Averaged Correlations between the True and Estimated Source', fontweight='bold', fontsize=16)
        plt.title(r'$\mathbf{S}_{\mathrm{X}}$', fontsize=30)

        if len(avg_JBSS_x) == 4:
            plt.yticks(np.arange(0, len(avg_JBSS_x)), labels=np.arange(1, len(avg_JBSS_x) + 1), fontsize=28)
            plt.xticks(np.arange(0, len(avg_JBSS_x[0])), labels=np.arange(1, len(avg_JBSS_x[0]) + 1), fontsize=28)
        else:
            plt.yticks(np.arange(0, len(avg_JBSS_x)), labels=np.arange(1, len(avg_JBSS_x) + 1), fontsize=22)
            plt.xticks(np.arange(0, len(avg_JBSS_x[0])), labels=np.arange(1, len(avg_JBSS_x[0]) + 1), fontsize=22)

        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False)
        full_path = path + '/' + 'JBSS1.png'

        if len(avg_JBSS_x) == 4:
            for i in range(len(avg_JBSS_x)):
                for j in range(len(avg_JBSS_x[0])):
                    c = np.around(avg_JBSS_x[j, i], 2)
                    txt = ax.text(i, j, str(c), va='center', ha='center', color='black', size='x-large')
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

        plt.tight_layout()
        plt.savefig(full_path)
        plt.show(block=False)

        font = {'style': 'italic',
                'weight': 'extra bold',
                'size': 20}

        mpl.rc('font', **font)

        fig, ax = plt.subplots()
        legend = ax.matshow(avg_JBSS_y, cmap='Blues')
        clrbr = plt.colorbar(legend, orientation="horizontal", pad=0.03)
        for t in clrbr.ax.get_xticklabels():
            t.set_fontsize(20)
        legend.set_clim(0, 1)
        clrbr.set_label(r'Correlation', fontsize=18)
        plt.ylabel(r'$\hat{\mathbf{\omega}}$', fontsize=40)
        #plt.suptitle('Averaged Correlations between the True and Estimated Source', fontweight='bold', fontsize=16)
        plt.title(r'$\mathbf{S}_{\mathrm{Y}}$', fontsize=30)
        if len(avg_JBSS_x) == 4:
            plt.yticks(np.arange(0, len(avg_JBSS_x)), labels=np.arange(1, len(avg_JBSS_x) + 1), fontsize=28)
            plt.xticks(np.arange(0, len(avg_JBSS_x[0])), labels=np.arange(1, len(avg_JBSS_x[0]) + 1), fontsize=28)
        else:
            plt.yticks(np.arange(0, len(avg_JBSS_x)), labels=np.arange(1, len(avg_JBSS_x) + 1), fontsize=22)
            plt.xticks(np.arange(0, len(avg_JBSS_x[0])), labels=np.arange(1, len(avg_JBSS_x[0]) + 1), fontsize=22)
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False)
        full_path = path + '/' + 'JBSS2.png'

        if len(avg_JBSS_x) == 4:
            for i in range(len(avg_JBSS_y)):
                for j in range(len(avg_JBSS_y[0])):
                    c = np.around(avg_JBSS_y[j, i], 2)
                    txt = ax.text(i, j, str(c), va='center', ha='center', color='black', size='x-large')
                    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

        plt.tight_layout()
        plt.savefig(full_path)
        plt.show(block=False)
        plt.close('all')
        # MAX JBSS
