#!/usr/bin/env python2

import numpy as np
import scipy.io

import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from mpl_toolkits import mplot3d

import os, sys

import argparse

from optimize_class import Fminsearch
from cluster_class import Clustering

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Find representative samples by clustering and optimize transform.')
    parser.add_argument('data_path', metavar='inpath', type=str, nargs='+',
                        help='path to mat file with dataset')
    parser.add_argument('outpath', metavar='outpath', type=str, nargs='+',
                        help='path to folder for cluster centers and image')
    
    parser.add_argument('vn', metavar='varname', type=str, help='variable name in the provided input file')
    

    args = parser.parse_args()
    
    print args 

    input_files = args.data_path
    outpath = args.outpath[0]

    print input_files
    print outpath


    print len(input_files)
    
    for fp in input_files:
        if not os.path.exists(fp):
            print "File not found"
            exit
    if len(input_files) == 1:
        if os.path.isdir(input_files[0]):
            prepath = os.path.abspath(input_files[0]) + '/'
            input_files = [prepath + i for i in os.listdir(input_files[0])]

    if not os.path.exists(outpath):
        print "Out folder not found. Creating folder ", outpath
        os.mkdir(os.path.abspath(outpath))


    n_clusters = 10

    xs = []
    fs = []

    for (i_fig, i_file) in enumerate(input_files):

        (dirpath, i_filename) = os.path.split(i_file)
        dirpath = os.path.abspath(dirpath)
        outpath = os.path.abspath(outpath)

        clustering = Clustering(dirpath, outpath, i_filename, 
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


