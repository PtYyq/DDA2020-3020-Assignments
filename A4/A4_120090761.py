# import library we need
import numpy as np
import pandas as pd
import time

# read and split the data
df = pd.read_table('seeds_dataset.txt', sep='\s+', header=None)
data = np.array(df)
X = data[:, 0:7]#features
label_true = data[:, 7]


# Part 1
# kmeans


class kmeans():
    def __init__(self, nclusters, random=False):
        self.nclusters = nclusters
        self.random = random

    def fit(self, data):
        t1 = time.time()
        #initialization
        if self.random:
            rng = np.random.RandomState()
        else:
            rng = np.random.RandomState(12)
        i = rng.permutation(data.shape[0])[:self.nclusters]
        centers = data[i]
        C = np.zeros([self.nclusters, data.shape[1]])
    # iterates until converges
        iterations = 0
        while True:
            iterations += 1
            # assign
            labels = []
            for i in range(data.shape[0]):
                dist = []
                for j in range(self.nclusters):
                    dist.append(np.linalg.norm(data[i]-centers[j]))          
                labels.append(np.argmin(dist)+1)# Add 1 since the true label start from '1'
            # get new centers
            for i in range(self.nclusters):
                c = np.mean(data[np.array(labels) == (i+1)],
                            axis=0)  # center for this cluster
                C[i] = c.copy()
            if (C == centers).all():
                break
            centers = C.copy()
        t2 = time.time()
        self.time = t2 - t1
        self.iterations = iterations
        self.centers = centers
        return centers

    def predict(self, data):
        labels = []
        for i in range(data.shape[0]):
            dist = []
            for j in range(self.nclusters):
                dist.append(np.linalg.norm(data[i]-self.centers[j]))
            labels.append(np.argmin(dist)+1)# Add 1 since the true label start from '1'
        return np.array(labels)


clu1 = kmeans(3)
clu1.fit(X)
print("Kmeans output labels:")
print(clu1.predict(X))

# acclerated kmeans


class accelerated_kmeans():
    def __init__(self, nclusters, random=False):
        self.nclusters = nclusters
        self.random = random

    def fit(self, data):
        t1 = time.time()
        if self.random:
            rng = np.random.RandomState()
        else:
            rng = np.random.RandomState(12)
        i = rng.permutation(data.shape[0])[:self.nclusters]
        centers = data[i]
        labels = []
        lxc = []
        ux = []
        for i in range(data.shape[0]):
            label = 0
            lc = [0 for i in range(self.nclusters)]
            for j in range(self.nclusters):
                if j == 0:
                    label = 1
                    dxc = np.linalg.norm(data[i]-centers[j])
                    lc[j] = dxc
                else:
                    if np.linalg.norm(centers[label-1]-centers[j]) < 2*np.linalg.norm(data[i]-centers[label-1]):
                        dxc_new = np.linalg.norm(data[i]-centers[j])
                        lc[j] = dxc_new
                        if dxc > dxc_new:
                            dxc = dxc_new
                            label = j + 1
            labels.append(label)
            lxc.append(lc)
            ux.append(dxc)
        r = [True for i in range(data.shape[0])]
        iterations = 0
        # Repeat 1-7 until convergence
        while True:
            iterations += 1
            # 1.
            sc = []
            dcc = np.zeros((self.nclusters, self.nclusters))
            for i in range(self.nclusters):
                for j in range(i+1, self.nclusters):
                    dcc[i][j] = np.linalg.norm(centers[i]-centers[j])
                    dcc[j][i] = dcc[i][j]
            for i in range(self.nclusters):
                d = []
                for j in range(self.nclusters):
                    if i != j:
                        d.append(np.linalg.norm(centers[i]-centers[j]))
                sc.append(0.5*min(d))
            for i in range(data.shape[0]):
                # 2.
                if ux[i] <= sc[labels[i]-1]:
                    pass
                # 3.
                else:
                    for j in range(self.nclusters):
                        # 3.
                        if (j != labels[i]-1) and (ux[i] > lxc[i][j]) and (ux[i] > 0.5*dcc[labels[i]-1][j]):
                            # 3a.
                            if r[i]:
                                dx_cx = np.linalg.norm(
                                    data[i]-centers[labels[i]-1])
                                ux[i] = dx_cx
                                r[i] = False
                            else:
                                dx_cx = ux[i]
                                # 3b.
                            if (dx_cx > lxc[i][j]) or (dx_cx > 0.5*dcc[labels[i]-1][j]):
                                dxc = np.linalg.norm(data[i]-centers[j])
                                lxc[i][j] = dxc
                                if dxc < dx_cx:
                                    labels[i] = j+1
                                    ux[i] = dxc
                # 4.
            mc = []
            for j in range(self.nclusters):
                mc.append(np.mean(data[np.array(labels) == (j+1)], axis=0))
            mc = np.array(mc)  # to np array
            for i in range(data.shape[0]):
                # 5.
                for j in range(self.nclusters):
                    lxc[i][j] = max(
                        0, lxc[i][j]-np.linalg.norm(centers[j]-mc[j]))
                # 6.
                ux[i] = ux[i] + \
                    np.linalg.norm(mc[labels[i]-1]-centers[labels[i]-1])
                r[i] = True
            if (mc == centers).all():
                break
            # 7
            centers = mc.copy()
        t2 = time.time()
        self.time = t2 - t1
        self.iterations = iterations
        self.centers = centers
        return centers

    def predict(self, data):
        labels = []
        for i in range(data.shape[0]):
            dist = []
            for j in range(self.nclusters):
                dist.append(np.linalg.norm(data[i]-self.centers[j]))
            # Add 1 since the true label start from '1'
            labels.append(np.argmin(dist)+1)
        return np.array(labels)


clu2 = accelerated_kmeans(3)
clu2.fit(X)
print("Accelerated Kmeans output labels:")
print(clu2.predict(X))


# GMM
class GMM():
    def __init__(self, nclusters, tolerance=1e-6, random=False):
        self.nclusters = nclusters
        self.tolerance = tolerance
        self.random = random

    def gaussian(self, x, mean, cov):
        left = 1/(pow((2*np.pi), 0.5*x.shape[0])*pow(np.linalg.det(cov), 0.5))
        right = np.exp(-0.5 * (x-mean).dot(np.linalg.inv(cov)).dot(x-mean))
        return left*right

    def initialization(self, data):
        # use k mean to get the initial weights, means and covariances
        clu = accelerated_kmeans(3, self.random)
        means = clu.fit(data)
        labels = np.array(clu.predict(data))
        weights = []
        covariances = []
        for j in range(self.nclusters):
            X = data[labels == j+1]
            weights.append(X.shape[0]/data.shape[0])
            u = 0
            for x in data:
                mul = x - means[j]
                u = u + \
                    np.matmul(mul.reshape(
                        data.shape[1], 1), mul.reshape(1, data.shape[1]))
            covariance = u/X.shape[0]
            covariances.append(covariance)
        return means, weights, covariances

    def fit(self, data):
        t1 = time.time()
        means, weights, covariances = self.initialization(data)
        # initial log likehood
        llh = 0
        for i in range(data.shape[0]):
            l = 0
            for j in range(self.nclusters):
                l += weights[j] * \
                    self.gaussian(data[i], means[j], covariances[j])
            llh += np.log(l)
        iterations = 0
        while True:
            iterations += 1
            # E-step:calculate gamma
            gamma = []
            for i in range(data.shape[0]):
                gk = []
                s = 0
                for j in range(self.nclusters):
                    s += weights[j] * \
                        self.gaussian(data[i], means[j], covariances[j])
                for j in range(self.nclusters):
                    value = weights[j] * \
                        self.gaussian(data[i], means[j], covariances[j])
                    gk.append(value/s)
                gamma.append(gk)
            # M-step
            # update mean-k
            for j in range(self.nclusters):
                u = 0
                for i in range(data.shape[0]):
                    u += gamma[i][j] * data[i]
                means[j] = u / np.sum(gamma, axis=0)[j]
            # update covariances
            for j in range(self.nclusters):
                e = 0
                for i in range(data.shape[0]):
                    mul = data[i] - means[j]
                    e = e + \
                        gamma[i][j]*np.matmul(mul.reshape(data.shape[1], 1),
                                              mul.reshape(1, data.shape[1]))
                    covariances[j] = e/np.sum(gamma, axis=0)[j]
            # update weights
                weights[j] = np.sum(gamma, axis=0)[j] / data.shape[0]
            # update log likehood
            new_llh = 0
            for i in range(data.shape[0]):
                l = 0
                for j in range(self.nclusters):
                    l += weights[j] * \
                        self.gaussian(data[i], means[j], covariances[j])
                new_llh += np.log(l)

            if abs(new_llh - llh) < self.tolerance:
                break
            llh = new_llh
        t2 = time.time()
        self.time = t2 - t1
        self.iterations = iterations
        self.means = means
        self.weights = weights
        self.covariances = covariances
        return means, weights, covariances

    def predict(self, data):
        gamma = []
        for i in range(data.shape[0]):
            gk = []
            s = 0
            for j in range(self.nclusters):
                s += self.weights[j] * \
                    self.gaussian(data[i], self.means[j], self.covariances[j])
            for j in range(self.nclusters):
                value = self.weights[j] * \
                    self.gaussian(data[i], self.means[j], self.covariances[j])
                gk.append(value/s)
            gamma.append(gk)
        labels = []
        for g in gamma:
            labels.append(np.argmax(g)+1)
        return np.array(labels)


clu3 = GMM(3)
clu3.fit(X)
print('GMM-EM output labels:')
print(clu3.predict(X))


# Part 2
# Silhouette Coefficient
def Silhouette_Coefficient_One(data, labels, index):
    k = np.unique(labels).size
    a = 0
    b = 0
    label = labels[index]
    for i in range(k):
        d = 0
        X = data[np.array(labels) == i + 1]
        for x in X:
            d += np.linalg.norm(x - data[index])
        if label == i+1:
            d = d/(X.shape[0]-1)
            a = d
        else:
            d = d/X.shape[0]
            if b == 0:
                b = d
            elif d < b:
                b = d
    return (b-a)/max(a, b)


def Silhouette_Coefficient(data, labels):
    Silhouette_Coefficients = []
    for i in range(data.shape[0]):
        Silhouette_Coefficients.append(
            Silhouette_Coefficient_One(data, labels, i))
    return np.mean(Silhouette_Coefficients)

# Rand Index


def Rand_Index(label_predict, label_true):
    a = b = c = d = 0
    for i in range(len(label_predict)):
        for j in range(i+1, len(label_predict)):
            if label_predict[i] == label_predict[j] and label_true[i] == label_true[j]:
                a += 1
            if label_predict[i] != label_predict[j] and label_true[i] != label_true[j]:
                b += 1
            if label_predict[i] != label_predict[j] and label_true[i] == label_true[j]:
                c += 1
            if label_predict[i] == label_predict[j] and label_true[i] != label_true[j]:
                d += 1
    return (a+b)/((a+b+c+d))


print("The Silhouette coefficient of Kmeans is:" +
      str(Silhouette_Coefficient(X, clu1.predict(X))))
print("The Rand Index of Kmeans is:" +
      str(Rand_Index(clu1.predict(X), label_true)))
print("The Silhouette coefficient of accelerated Kmeans is:" +
      str(Silhouette_Coefficient(X, clu2.predict(X))))
print("The Rand Index of accelerated  Kmeans is:" +
      str(Rand_Index(clu2.predict(X), label_true)))
print("The Silhouette coefficient of GMM-EM is:" +
      str(Silhouette_Coefficient(X, clu3.predict(X))))
print("The Rand Index of GMM-EM is:" +
      str(Rand_Index(clu3.predict(X), label_true)))

# Part3


def variance(scores):
    s = 0
    mean = np.mean(scores)
    for score in scores:
        s += (score - mean)**2
    return s/len(scores)


def sensitivity(model, data, times):
    se = []
    ri = []
    for i in range(times):
        model.fit(data)
        predict = model.predict(data)
        se.append(Silhouette_Coefficient(data, model.predict(data)))
        ri.append(Rand_Index(predict, label_true))
    se_variance = variance(se)
    ri_variance = variance(ri)
    return se_variance, ri_variance


# Since running this procudure cost a lot of time, I set False here, if you want to run it, set it as True
calculateSensitivity = False
if calculateSensitivity:
    se1, ri1 = sensitivity(kmeans(3, random=True), X, 10)
    print("Variance of Silhouette coefficient of Kmeans is:"+str(se1))
    print("Variance of Rand Index of Kmeans is:"+str(ri1))
    se2, ri2 = sensitivity(accelerated_kmeans(3, random=True), X, 10)
    print("Variance of Silhouette coefficient of accelerated Kmeans is:"+str(se2))
    print("Variance of Rand Index of accelerated Kmeans is:"+str(ri2))
    se3, ri3 = sensitivity(GMM(3, random=True), X, 10)
    print("Variance of Silhouette coefficient of GMM-EM is:"+str(se3))
    print("Variance of Rand Index of GMM-EM is:"+str(ri3))


# Part 4
print("The iterations of kmeans is:"+str(clu1.iterations))
print("The required time of kmeans is:"+str(clu1.time))
print("The iterations of accelerated kmeans is:"+str(clu2.iterations))
print("The required time of accelerated kmeans is:"+str(clu2.time))
print("The iterations of GMM-EM is:"+str(clu3.iterations))
print("The required time of GMM-EM is:"+str(clu3.time))
