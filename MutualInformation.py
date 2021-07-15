import matplotlib.pyplot as plt
import scipy.spatial as ss
from scipy.special import digamma, gamma
import numpy as np
from scipy import ndimage
import math
from math import log, pi, hypot, fabs, sqrt

def MutualInformation(mu_x, mu_y, sigma_x, sigma_y, rho):
    f_x = lambda x: 1 / (np.sqrt(2 * np.pi) * sigma_x) * np.exp(-0.5 * ((x - mu_x) / sigma_x) ** 2)
    f_y = lambda y: 1 / (np.sqrt(2 * np.pi) * sigma_y) * np.exp(-0.5 * ((y - mu_y) / sigma_y) ** 2)
    f_xy = lambda x, y: 1 / (np.sqrt(1 - rho ** 2) * sigma_x * sigma_y * 2 * np.pi) * \
    np.exp(-0.5 * 1/(1 - rho ** 2) * ((((x - mu_x) / sigma_x) ** 2) + (((y - mu_y) / sigma_y) ** 2) - 2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y)))

    t = np.linspace(-10, 10, 1000)
    plt.scatter(t, f_x(t), marker='.', s=10, color='black')
    plt.scatter(t, f_y(t), marker='.', s=10, color='red')
    plt.show()

class MutualInfo:

    @staticmethod
    def zip2(*args):
        # zip2(x,y) takes the lists of vectors and makes it a list of vectors in a joint space
        # E.g. zip2([[1],[2],[3]],[[4],[5],[6]]) = [[1,4],[2,5],[3,6]]
        return [sum(sublist, []) for sublist in zip(*args)]

    @staticmethod
    def avgdigamma(points, dvec):
        # This part finds number of neighbors in some radius in the marginal space
        # returns expectation value of <psi(nx)>
        N = len(points)
        tree = ss.cKDTree(points)
        avg = 0.
        for i in range(N):
            dist = dvec[i]
            # subtlety, we don't include the boundary point,
            # but we are implicitly adding 1 to kraskov def bc center point is included
            num_points = len(tree.query_ball_point(points[i], dist - 1e-11, p=float('inf')))
            avg += digamma(num_points) / N
            print(f'Avg {avg}')
        return avg

    @staticmethod
    def mi_Kraskov(X, k=5):
        isNaN = True
        while(isNaN):
            X[0] = X[0] + np.random.normal(0, 0.1, (np.array(X).shape[1])).T
            print(f'Shape of X: {np.array(X).shape}')

            # adding small noise to X, e.g., x<-X+noise
            x = []
            for i in range(len(X)):
                tem = []
                for j in range(len(X[i])):
                    tem.append([X[i][j]])
                x.append(tem)

            points = []
            for j in range(len(x[0])):
                tem = []
                for i in range(len(x)):
                    tem.append(x[i][j][0])
                points.append(tem)
            tree = ss.cKDTree(points)

            dvec = []
            for i in range(len(x)):
                dvec.append([])

            for point in points:
                # Find k-nearest neighbors in joint space, p=inf means max norm
                knn = tree.query(point, k + 1, p=float('inf'))
                points_knn = []
                for i in range(len(x)):
                    dvec[i].append(float('-inf'))
                    points_knn.append([])
                for j in range(k + 1):
                    for i in range(len(x)):
                        points_knn[i].append(points[knn[1][j]][i])

                # Find distances to k-nearest neighbors in each marginal space
                for i in range(k + 1):
                    for j in range(len(x)):
                        if dvec[j][-1] < fabs(points_knn[j][i] - points_knn[j][0]):
                            dvec[j][-1] = fabs(points_knn[j][i] - points_knn[j][0])

            ret = 0.
            for i in range(len(x)):
                print(f'intermediate result: {ret}')
                ret -= MutualInfo.avgdigamma(x[i], dvec[i])
            ret += digamma(k) - (float(len(x)) - 1.) / float(k) + (float(len(x)) - 1.) * digamma(len(x[0]))

            if ret < 0:
                ret = 0

            norm_mi = np.sqrt(1 - np.exp(-0.5 * ret))

            if math.isnan(norm_mi):
                isNaN = True
            else:
                isNaN = False

        return norm_mi

    @staticmethod
    def mutual_information_2d(x, y, sigma=1, normalized=False):
        """
        Computes (normalized) mutual information between two 1D variate from a
        joint histogram.
        Parameters
        ----------
        x : 1D array
            first variable
        y : 1D array
            second variable
        sigma: float
            sigma for Gaussian smoothing of the joint histogram
        Returns
        -------
        nmi: float
            the computed similariy measure
        """
        EPS = np.finfo(float).eps

        bins = (256, 256)

        jh = np.histogram2d(x, y, bins=bins)[0]

        # smooth the jh with a gaussian filter of given sigma
        ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                output=jh)

        # compute marginal histograms
        jh = jh + EPS
        sh = np.sum(jh)
        jh = jh / sh
        s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
        s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

        # Normalised Mutual Information of:
        # Studholme,  jhill & jhawkes (1998).
        # "A normalized entropy measure of 3-D medical image alignment".
        # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
        if normalized:
            mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                  / np.sum(jh * np.log(jh))) - 1
        else:
            mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
                  - np.sum(s2 * np.log(s2)))

        norm_mi = np.sqrt(1-np.exp(-0.5*mi))

        return norm_mi

