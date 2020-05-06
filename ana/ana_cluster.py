#!/usr/bin/env python2

import sklearn.cluster as sklcl
import numpy as np
import scipy.io

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib._color_data as mcd

import os, sys


def mean(a):
    return sum(a) / len(a)

class Clustering:

    def __init__(self, filename, outpath, *args, **kwargs):

        mat = scipy.io.loadmat(filename)
        self.data = mat['data']

        if len(kwargs) > 0:
            if 'method' in kwargs.keys():
                self.methodname = kwargs['method']

                if 'KMeans' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.KMeans(n_clusters = n['n_clusters'],random_state=0)
                    else:
                        self.method = sklcl.KMeans(random_state=0)

                if 'DBSCAN' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.DBSCAN(eps=n['eps'], min_samples=n['min_samples'])
                    else:
                        self.method = sklcl.DBSCAN()

                if 'OPTICS' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.OPTICS(max_eps=n['max_eps'], min_samples=n['min_samples'])
                    else:
                        self.method = sklcl.OPTICS()


                if 'AffinityPropagation' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.AffinityPropagation(damping=n['damping'])
                    else:
                        self.method = sklcl.AffinityPropagation()


                if 'MeanShift' in self.methodname:
                    if 'xtra' in kwargs.keys():
                        n = kwargs['xtra']
                        self.method = sklcl.MeanShift(bandwidth=n['bandwidth'])
                    else:
                        self.method = sklcl.MeanShift()


                else:
                    pass
        else:
            print "clustering with KMeans"
            self.method = sklcl.KMeans(random_state=0)
            self.methodname = 'KMeans'


        self.filename = filename
        self.outpath = outpath
        title = self.filename.split('.mat')
        self.title = title[0]

        self.centres = []

        

    def get_centres(self):
        return self.centres


    def do_cluster(self):
        self.method.fit(self.data)
        self.labels = self.method.labels_
        try:
            self.centres = self.method.cluster_centers_
        except:
            print "no centres in method", self.methodname
        self.lset = set(self.labels)



    def plot_result(self, i_fig):

        fig = plt.figure(i_fig)
        ax1 = plt.subplot(2,1,1, projection='3d')
        ax2 = plt.subplot(2,1,2, projection='3d')

        color_keys = mcd.CSS4_COLORS.keys()

        for label, color in zip(self.lset, color_keys):
        
            indices = [i for i, x in enumerate(self.labels) if x == label]
            cluster_members = np.asarray([self.data[i] for i in indices])

            cn = mcd.CSS4_COLORS[color]

            cm = cluster_members
            ax1.scatter(cm[:,0], cm[:,1], cm[:,2], color=cn)

            if len(indices) > 5:
                if not label == -1: 
                    ax2.scatter(cm[:,0], cm[:,1], cm[:,2], color=cn)
    

        plt.suptitle(self.methodname)



    def save_plot(self):
        plt.savefig(self.outpath+self.methodname+'.png')


    def save_to_mat(self):

        name = self.outpath + "clusters_" + self.methodname + ".mat"
        scipy.io.savemat(name, {'labels':self.labels, 'data':self.data})



if __name__ == '__main__':
    
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        outpath = sys.argv[2]
    else:
        path = "/home/marsela/specularia/gith/ana/ana_pcl.mat"
        outpath = "/home/marsela/specularia/gith/ana/"


    methods = [ 'KMeans', 
                'DBSCAN', 
                # 'OPTICS',
                'AffinityPropagation',
                'MeanShift']
    params = [  {'n_clusters':20},
                {'eps':2,'min_samples':5},
                # {'max_eps':np.inf,'min_samples':5},
                {'damping':0.95},
                {'bandwidth':None} ]
    i_fig = 1
    for (method, param) in zip(methods, params):
        i_filename = path

        clustering = Clustering(path, outpath, 
                                method=method, xtra=param)
        clustering.do_cluster()
        clustering.plot_result(i_fig)

        print "---------"
        print method
        print clustering.lset
        lens = [len([i for i in clustering.labels if i==l]) for l in clustering.lset]
        print lens
        i_fig +=1

        clustering.save_plot()
        clustering.save_to_mat()

    plt.show()




    # def voxel_plot(self, i_fig):

    #     fig = plt.figure(i_fig)
    #     # ax = fig.gca(projection='3d')
    #     ax = plt.subplot(2,1,2, projection='3d')

    #     data = self.data

    #     data[:,0] -= np.min(data[:,0])
    #     data[:,1] -= np.min(data[:,1])
    #     data[:,2] -= np.min(data[:,2])

    #     # prepare some coordinates
    #     x, y, z = np.indices((int(np.max(data[:,0])), int(np.max(data[:,1])), int(np.max(data[:,2]))))

    #     # print x
    #     cubes = []
    #     colors = []
    #     color_keys = mcd.CSS4_COLORS.keys()
    #     for label, center, color in zip(self.lset, self.centres, color_keys):
    #         indices = [i for i, x in enumerate(self.labels) if x == label]
    #         cluster_members = np.asarray([data[i] for i in indices])
    #         cn = mcd.CSS4_COLORS[color]
    #         c = [int(i) for i in cluster_members[0]]
    #         cube =  (x == int(c[0])) & (y == int(c[1])) & (z == int(c[2]))
    #         for c in cluster_members:
    #             c = [int(i) for i in c]
    #             cube = cube | (x == int(c[0])) & (y == int(c[1])) & (z == int(c[2]))

    #         cubes.append(cube)
    #         colors = np.empty(cube.shape, dtype=object)
    #         # colors[cube] = cn

    #     # combine the objects into a single boolean array
    #     voxels = cubes[0]
    #     # print cubes
    #     for cube,col in zip(cubes, colors):
    #         voxels = voxels | cube

    #     # colors = np.empty(voxels.shape, dtype=object)
    #     # for cube, color in zip(cubes, color_keys):
    #     #     cn = mcd.CSS4_COLORS[color]
    #     #     colors[cube] = 'red'
            

    #     # # set the colors of each object
    #     colors = np.empty(voxels.shape, dtype=object)
    #     for cube, color in zip(cubes, color_keys):
    #         cn = mcd.CSS4_COLORS[color]
    #         colors[cube] = cn

    #     print np.sum(voxels)

    #     # and plot everything
    #     ax.voxels(voxels, facecolors=colors, edgecolor='k')

    #     plt.show()