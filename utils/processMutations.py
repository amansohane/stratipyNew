# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:57:50 2015

@author: aman
"""

import scipy.sparse as sp
import pickle
import numpy
import random
import utils
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram

alpha = 0.7
mutations_min = 6
diff_thresh = 10**-6
nclust = 3
maxiter = 10
tolerance = 0.01
gamma = 0

f = open('../res/network.data','r')
data = pickle.load(f)
genes = data['genes']
f.close()

##patient x genes
##for unit testing
mutations = numpy.random.random_integers(0,1,(100,10))
thisgenes = random.sample(genes,10)
##

toadd = [i for i in genes if i not in thisgenes]
tokeep = [i for i,j in enumerate(thisgenes) if j in genes]

mutations = mutations[mutations.sum(axis=1) > mutations_min]
mutations_temp = numpy.zeros((len(mutations),len(genes)))
mutations_temp[:,tokeep] = mutations[:,tokeep]
mutations = sp.csr_matrix(mutations_temp)

mutation_smooth = utils.diffuse(mutations,data['adj'],alpha,diff_thresh)
#quantile normalisation

#U,V = utils.gnmf(mutation_smooth,data['knn'],nclust, gamma, maxiter, tolerance)
#labels = numpy.array(V.todense().argmax(axis=1))[:,0]

def gnmfsingle(X, W, nclust, gamma, maxiter, tolerance):
    U,V = utils.gnmf(X, W ,nclust, gamma, maxiter, tolerance)
    return numpy.array(V.todense().argmax(axis=1))[:,0]

cons = utils.consensus(gnmfsingle,mutation_smooth, [data['knn'],nclust, gamma, maxiter, tolerance], bootstrap = 0.8,rep = 10)

######take from stratipy modules
#zmatrix = linkage(cons)
#clusters = fcluster(zmatrix,1)
#dend = dendrogram(zmatrix,count_sort='ascending')

