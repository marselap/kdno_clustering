#!/usr/bin/env python2
import numpy as np
import scipy.optimize
import scipy
import scipy.io
from tf.transformations import quaternion_matrix
import time
import sys
import random

class Fminsearch:
    def __init__(self, *args):
        
        if len(args) > 0:
            self.passed_arg = args[0]
        else:
            self.passed_arg = []
        pass

    def get_data_matrix(self):
        read_dataset = self.read_dataset

        if (len(np.shape(read_dataset)) == 2) and 7 in np.shape(read_dataset):
            self.pos_quat = read_dataset
        elif (len(np.shape(read_dataset)) == 3) and 4 in np.shape(read_dataset):
            self.Txs = read_dataset.transpose(2,1,0)

        if self.Txs is None:
            [m, n] = np.shape(self.pos_quat)
            Txs = np.empty((m, 4, 4))#, order = 'F')
            a = np.empty((4,4))
            for i, row in enumerate(self.pos_quat):                        
                a = np.asmatrix(quaternion_matrix(row[3:]))
                a[0:3,3] = np.resize(row[0:3], (3,1))
                Txs[i,:,:] = a.transpose()
            self.Txs = Txs
        


    def load_set(self, *args):
        if len(args) > 0:
            self.passed_arg = args[0]

        self.Txs = None
        filename = None

        if len(self.passed_arg) > 0:
            if isinstance(self.passed_arg, str):
                filename = self.passed_arg
                if len(args) > 1:
                    var_name = args[1]
                else:
                    var_name = "data"
                
                mat = scipy.io.loadmat(filename)
                try:
                    read_dataset = mat[var_name]
                except KeyError as e:
                    print "no data under: ", var_name

            elif isinstance(self.passed_arg, np.ndarray):
                read_dataset = self.passed_arg

            elif isinstance(self.passed_arg, list):
                read_dataset = np.asarray(self.passed_arg)
            
        else:
            inpath = "/home/marsela/Documents/MATLAB/cluster/VU_exp_q/"
            i_filename = "VU_0406_EXP1_krug_vrtnja_3_caltx.mat"
            filename = inpath + i_filename
            var_name = "data"
            mat = scipy.io.loadmat(filename)
            try:
                read_dataset = mat[var_name]
            except KeyError as e:
                print "no data under: ", var_name

        self.read_dataset = read_dataset

        try:
            a = filename.split('/')
            a = a[-1]
            a = a.split('.mat')
            a = a[0]
            self.dataset_name = a
        except:
            print "could not parse dataset name"
            self.dataset_name = "dataset"
        self.get_data_matrix()

    def do_optimise_pos(self, p1, r1, txs = None):
        
        P1 = np.array(p1, dtype="float32")
        R = np.float32(r1).transpose()
        if txs is None:
            txs = np.float32(self.Txs[::,:,:])

        objectiveFunLambda = lambda x: self.objectiveFunPosition(x, R, txs)
        
        m,n,k = np.shape(txs)
        
        print("")
        print ("Start optimize transform on dataset " + self.dataset_name)
        print ("Optimizing transform on " +str(m)+ " data points.")
        print ("---------------")
        start = time.time()
        xopt = scipy.optimize.fmin(func=objectiveFunLambda, x0=P1)
        end = time.time()
        
        print ("---------------")
        print("Optimization duration: " + str(end - start) + " seconds.")
        print ("Optimized transform: " + str(xopt))
        print("")
        return xopt, self.objectiveFunPosition(xopt, R, txs)


    def do_optimise_pos_random(self, p1, r1, n_rand):
        

        m,n,k = np.shape(self.Txs)
        rand_indices = random.sample(range(0, m), n_rand)        

        txs = np.float32(self.Txs[rand_indices,:,:])
        xopt, fopt = self.do_optimise_pos(p1, r1, txs)

        return xopt, fopt


    def objectiveFunPosition(self, P, R, T0A):
        #Optimization for position - transposed matrices for faster computation

        Tx = np.eye(4, dtype="float32")
        Tx[0:3, 0:3] = R
        Tx[3, 0:3] = P

        [l, m, n] = np.shape(T0A); 
        f = 0
        for i in xrange(l):
            Told_1 = np.dot(Tx, T0A[i,:,:])
            for j in xrange(l):
                Told = np.dot(Tx, T0A[j,:,:])
                p = Told[3, 0:3] - Told_1[3, 0:3]
                f = f + np.linalg.norm(p)

        return f



if __name__ == "__main__":


    optim = Fminsearch()
    optim.load_set("/home/marsela/Documents/MATLAB/cluster/VU_eksperimenti/VU_0406_EXP1_krug_vrtnja_3.mat", "CalTx")
    
    p1 = [0,0,0]
    r1 = np.eye(3)

    optim.do_optimise_pos_random(p1,r1,10)

    pass


# Optimization terminated successfully.
#          Current function value: 1326.962737
#          Iterations: 124
#          Function evaluations: 253
# ('optimization duration: ', 1358.044676065445)
# ('print optimized transform: ', array([ 0.01155917, -0.15891211,  0.00319799]))


# def objectiveFunPosition1(P, R, T0A):
#     #Optimization for position

#     Tx = np.eye(4, dtype="float32")
#     Tx[0:3, 0:3] = R
#     Tx[3, 0:3] = P

#     [l, m, n] = np.shape(T0A); 
#     f = 0
#     for i in xrange(l):
#         for j in xrange(l):
#             a = np.dot(Tx, T0A[j,:,:]-T0A[i,:,:])
#             f = f + np.linalg.norm(a[3, 0:3])

#     # print f
#     return f



# def objectiveFunPositionOriginal(P, R, T0A):
#     #Optimization for position

#     Tx = np.eye(4)
#     Tx[0:3, 0:3] = R
#     Tx[0:3, 3] = P

#     [l, m, n] = np.shape(T0A); 
#     f=0; 
#     for i in xrange(l):
#         Told_1 = np.dot(T0A[i,:,:], Tx)
#         for j in xrange(l):
#             Told = np.dot(T0A[j,:,:], Tx)
#             p = Told[0:3, 3] - Told_1[0:3, 3]
#             f = f + np.linalg.norm(p)

#     # print f
#     return f