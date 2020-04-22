#!/usr/bin/env python2

# from sklearn.cluster import KMeans
import sklearn.cluster as sklcl
import numpy as np
import scipy.io

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib._color_data as mcd

from tf.transformations import quaternion_from_matrix, quaternion_matrix

import os, sys

from optimize_class import Fminsearch


def pos_q_2_pos_vec(row):
    s = []
    for i in [0,1,2]:
        s.append(row[i])
    
    x = row[3]
    y = row[4]
    z = row[5]
    w = row[6]
    V = [0, 0, 0]

    V[0] = 2 * (x * z - w * y)
    V[1] = 2 * (y * z + w * x)
    V[2] = 1 - 2 * (x * x + y * y)
    
    V = np.array(V)
    V /= 100*np.linalg.norm(V)
    for i in [0,1,2]:
        s.append(V[i])

    return s


def find_closest_member(cluster_members, center):
    min_dist = -1.
    min_index_subset = -1
    for i in xrange(len(cluster_members)):
        dist = np.linalg.norm(cluster_members[i, 0:3] - np.asarray(center))  
        if (min_dist == -1.) or dist < min_dist:
            min_dist = dist
            min_index_subset = i
    return min_index_subset


def mean(a):
    return sum(a) / len(a)

class Clustering:

    def __init__(self, inpath, outpath, filename, *args, **kwargs):

        fullpath = os.path.join(inpath,filename)
        mat = scipy.io.loadmat(fullpath)

        self.pos_quat = self.get_data_pos_quat(mat)
        self.pos_dir_vec = np.array([pos_q_2_pos_vec(row) for row in self.pos_quat])

        tmpdict = {}
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                tmpdict[key] = value
        else:
            print "clustering to 10 clusters"
            self.n_clusters = 10

        if 'n_clusters' in tmpdict.keys():
            self.n_clusters =  tmpdict['n_clusters']
        else:
            self.n_clusters = 10

        self.method = sklcl.KMeans(n_clusters=self.n_clusters, random_state=0)

        self.filename = filename
        self.outpath = outpath
        title = self.filename.split('.mat')
        self.title = title[0]

        self.centres_q = []
        self.centres_dirvec = []


    def get_data_pos_quat(self, mat):

        pos_quat = None
        try:
            pos_quat = mat['data']
        except KeyError:
            try:
                caltx = mat['CalTx']
            except KeyError:
                print "nema ni data ni CalTx. Ne podrzavamo Tp_rec i ostalo"

        if pos_quat is None:
            m = max(np.shape(caltx))
            pos_quat = np.empty((m,7))
            
            a = np.empty((1, 7))

            caltx = np.transpose(caltx, (2,0,1))

            for i, mtrx in enumerate(caltx):
                q = quaternion_from_matrix(mtrx)
                p = mtrx[0:3,3].transpose()
                # print np.shape(a[0:3])
                # a[1,0:3] = np.resize(p,(1,3))
                a[0,0:3] = p
                a[0,3:] = q
                pos_quat[i,:] = a

        return pos_quat

    def do_cluster(self):
        self.method.fit(self.pos_quat[:,0:3])
        self.labels = self.method.labels_
        self.centres = self.method.cluster_centers_
        self.lset = set(self.labels)

    def add_missing_values(self):
        #hard coded center = closest_point(center)
        self.centres_q = []
        self.centres_dirvec = []
        for label, center in zip(self.lset, self.centres):
        
            indices = [i for i, x in enumerate(self.labels) if x == label]
            cluster_members = np.asarray([self.pos_dir_vec[i] for i in indices])

            min_index_subset = find_closest_member(cluster_members, center)
            min_index_all = indices[min_index_subset]
            
            centre_q = self.pos_quat[min_index_all]
            centre_dirvec = cluster_members[min_index_subset]

            self.centres_q.append(centre_q)
            self.centres_dirvec.append(centre_dirvec)
        
        self.centres_q = np.asarray(self.centres_q)
        self.centres_dirvec = np.asarray(self.centres_dirvec)
    


    def plot_result(self, i_fig):

        if self.centres_q == []:
            self.add_missing_values()

        fig = plt.figure(i_fig)
        plt.title(self.title)
        ax1 = plt.subplot(2,1,1, projection='3d')
        ax2 = plt.subplot(2,1,2, projection='3d')

        color_keys = mcd.CSS4_COLORS.keys()

        for label, center, color in zip(self.lset, self.centres_dirvec, color_keys):
        
            indices = [i for i, x in enumerate(self.labels) if x == label]
            cluster_members = np.asarray([self.pos_dir_vec[i] for i in indices])

            cn = mcd.CSS4_COLORS[color]

            cm = cluster_members
            ax1.quiver(cm[:,0], cm[:,1], cm[:,2], cm[:,3], cm[:,4], cm[:,5], color=cn)

            i = center    
            ax2.quiver(i[0], i[1], i[2], i[3], i[4], i[5], color=cn)
    
    def save_plot(self):
        plt.savefig(self.outpath+self.title)

    def save_to_mat(self):

        if self.centres_q == []:
            self.add_missing_values()

        self.centres_t = []
        for c in self.centres_q:
            t = np.asmatrix(quaternion_matrix(c[3:]))
            t[0:3, 3] = np.reshape(c[0:3], (3,1))
            self.centres_t.append(t)

        self.centres_t = np.asarray(self.centres_t)
        self.centres_t = np.reshape(self.centres_t, (1,2,0))
        name = self.outpath + "centres_" + self.title + "_" + str(self.n_clusters) + ".mat"
        # scipy.io.savemat(name, {'c_c':self.centers})
        scipy.io.savemat(name, {'c_c':self.centres_t})



if __name__ == '__main__':
    
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        outpath = sys.argv[2]
    else:
        path = "/home/marsela/Documents/MATLAB/cluster/VU_exp_q/"
        outpath = "/home/marsela/Documents/MATLAB/cluster/VU_exp_q_centers/"

    all_files = os.listdir(path)
    caltx_files = [i for i in all_files if 'caltx' in i]

    krug_vrtnja = [i for i in caltx_files if "krug_vrtnja_2" in i]

    n_clusters = 10

    xs = []
    fs = []
    for n_clusters in xrange(5,20):
        i_fig = 1
        i_filename = krug_vrtnja[0]

        # for (i_fig, i_filename) in enumerate(krug_vrtnja):
        clustering = Clustering(path, outpath, i_filename, 
                                method="kmeans", n_clusters = n_clusters)
        clustering.do_cluster()
        clustering.add_missing_values()


        optim = Fminsearch()
        optim.load_set(clustering.centres_q)
        p1 = [0,0,0]
        r1 = np.eye(3)

        xopt, fopt = optim.do_optimise_pos(p1,r1)
        xs.append([xopt])
        fs.append([fopt])

    benchmark = np.asarray([ 0.01155917, -0.15891211,  0.00319799])

    xplot = []
    yplot = []
    for x,n in zip(xs, xrange(5,20)):
        print "clusters: ", n
        xplot.append(n)
        print np.linalg.norm(benchmark - np.asarray(x))
        yplot.append(np.linalg.norm(benchmark - np.asarray(x)))


    fig = plt.figure()
    plt.subplot(211)
    plt.scatter(xplot, yplot)
    plt.grid("on")
    plt.title('dist to bench')

    plt.subplot(212)
    plt.scatter(xplot, fs)
    plt.grid("on")
    plt.title('f_opt')
    
    plt.show()

    #     clustering.plot_result(i_fig)
    #     clustering.save_plot()
    #     clustering.save_to_mat()
    
    # plt.show()


