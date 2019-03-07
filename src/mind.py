# -*- coding: utf-8 -*-
"""
Implementation of MIND algorithm (Low&Lewallen'18) 
* Fit trees
* find_node
* compute_prob
* Learn coordinates

TO DO (efficiency): - splitting (line search, 1 PCA per direction)
- landmark approach
- preprocessing: PCA
- inheritance
Created on Mon Nov 19 10:35:31 2018
@author: joram
"""
import autograd.numpy as np
from autograd import grad
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
from scipy.sparse.csgraph import shortest_path # global distances
from scipy.optimize import minimize # intrinsic coordinates
from sklearn.manifold import MDS # init for intrinsic coordinate optim

class mind_ensemble():
    """ Ensemble of trees """    
    
    def __init__(self, x, manifold_dim=None, n_trees=1, seed=123):
        self.x = x
        self.manifold_dim = manifold_dim
        self.n_trees = n_trees
        self.trees = np.zeros((n_trees,), dtype=object)
        self.P = None
        self.D = None
        self.rng = np.random.RandomState(seed)
        self.y = None
        
    def learn_coordinates(self):
        """ Learn intrinsic coordinates that minimize loss_fun """
        # initialize using MDS
        if self.D is None:
            self.compute_distances()
        print("Initialize using MDS")
        mds = MDS(n_components=self.manifold_dim, dissimilarity='precomputed')
        y_init = mds.fit_transform(self.D)
        loss_grad = grad(self.loss_fun)
        print("Perform optimization")
        res = minimize(self.loss_fun, x0=y_init.flatten(), jac=loss_grad)
        print(res['message'])
        y_final = res['x'].reshape((y_init.shape))
        # Rotate &  whiten to obtain canonical representation
        pca = PCA(whiten=True)
        self.y = pca.fit_transform(y_final)
        
    def loss_fun(self, y):
        """ Loss function for intrinsic coordinates y
            y: (n, dim) """
        if self.D is None:
            self.compute_distances
        eps = 1e-100
        D = self.D
        n = D.shape[0]
        y = y.reshape((n,self.manifold_dim))
        dif = np.sqrt(np.sum((y[:,np.newaxis] - y[np.newaxis,:])**2, axis=-1)+eps)
        error = 1/(D+np.eye(n)) * (D-dif)**2 # diag prevents 1/0, not counted in triu
        dist = np.sum(np.triu(error, k=1))
        return dist
        
    def compute_probs(self):
        """ Compute prob as median over all trees """
        T = self.x.shape[0]        
        self.single_P = np.zeros((self.n_trees, T, T))
        for t in range(self.n_trees):
            tree = mind_tree(self.x, manifold_dim=self.manifold_dim, rng=self.rng)
            self.trees[t] = tree
            tree.compute_probs()
            self.single_P[t] = tree.P
        # un-normalized density
        self.P = np.median(self.single_P,axis=0) 
        
    def compute_distances(self):
        """ Compute distances from probabilities
            TO DO: fully vectorize """
        if self.P is None:
            self.compute_probs()
        eps = 1e-100
        D = np.sqrt(-np.log(self.P)+eps)
        D = shortest_path(D,method='J') # global distances
        self.D = (D + D.T)/2 # symmetrize
        
        
class mind_tree():
    """ Single tree"""    
    
    def __init__(self, x, manifold_dim=None, rng=None, seed=123):
        """ x: (T, N) with N neurons, T time 
            n_dir: random splitting directions
            n_leaf: minimum samples in a leaf node"""
        self.x = x
        if rng is None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = rng
        self.dim = x.shape[1]
        if manifold_dim is None:
            self.manifold_dim = self.dim
        else:
            self.manifold_dim = manifold_dim
        # methods will initialize when called
        self.n_dir = None
        self.n_leaf = None
        self.tree = None
        self.leaves = []
        self.D = None
        self.coordinates = None
        self.node_id = None
        # TO DO: check if None in functions, don't compute if not None
        
    def fit_tree(self, x=None,level=0, number=0, n_dir=10, n_leaf=40):
        """ Recursively partition state space by fitting decision trees
            n_dir: n. of random splitting directions
            n_leaf: minimum samples in a leaf node 
            start recursion: fit_tree(0,0)"""
        if x is None:
            x = self.x
        n = x.shape[0]  # samples
        if n < 2*n_leaf: # stop, fit pca to successors
            leaf = self.make_leaf(x, number, level)
            self.leaves.append(leaf)
            return leaf
        # Not done yet, determine split
        # generate random dirs 
        #hplanes = self.generate_hyperplanes(n_dir)
        scores  = np.zeros((n_dir,))
        v = self.rng.normal(0,1,size=(self.dim, n_dir))
        v /= np.sum(v**2, 0)**0.5
        c = self.rng.uniform(-1,1,size=n_dir)
        #for i, (v, c) in enumerate(zip(hplanes['directions'], hplanes['thresholds'])):
        for i in range(n_dir):        
            # split x according to hplane
            idx_left = np.dot(x, v[:,i]) < c[i] 
            # determine successors of samples left/right of plane
            successor_idx = np.where(idx_left)[0] + 1 
            successor_idx = successor_idx[successor_idx<n] 
            p = np.mean(idx_left)
            # fit Gaussians to successors
            if p*n > n_leaf and (1-p)*n > n_leaf:
                L = x[successor_idx]
                R = x[successor_idx]
                # score: weighted average of log likes single splits
                scores[i] = self.fit_ppca(L).ll*p + self.fit_ppca(R).ll*(1-p)
        # select best split
        i_opt = np.argmin(scores)
        v_opt = v[:,i_opt] #hplanes['directions'][i_opt]
        c_opt = c[i_opt] #hplanes['thresholds'][i_opt]
        idx_left = np.dot(x, v_opt) < c_opt
        # check if done
        if (np.sum(idx_left) < n_leaf) | (len(idx_left)-np.sum(idx_left)< n_leaf):
            leaf = self.make_leaf(x, number, level)
            self.leaves.append(leaf)
            return leaf
        # Not done, make splits and continue
        left_leaf = self.fit_tree(x[idx_left], level+1, number+1, 
                                   n_dir=n_dir, n_leaf=n_leaf)
        leaf_number = self.leaves[-1]['number']
        right_leaf = self.fit_tree(x[~idx_left], level+1, leaf_number+1, 
                                   n_dir=n_dir, n_leaf=n_leaf)
        split = {'type': 'split', 'level': level, 'number': number,
                 'v': v_opt, 'c': c_opt, 'left': left_leaf, 'right': right_leaf}
        if (level==0) & (number==0):
            self.tree = split
        return split
        
    def learn_coordinates(self):
        """ Learn intrinsic coordinates that minimize loss_fun """
        # initialize using MDS
        if self.D is None:
            self.compute_distances()
        print("Initialize using MDS")
        mds = MDS(n_components=self.manifold_dim, dissimilarity='precomputed')
        y_init = mds.fit_transform(self.D)
        loss_grad = grad(self.loss_fun)
        print("Perform optimization")
        res = minimize(self.loss_fun, x0=y_init.flatten(), jac=loss_grad)
        print(res['message'])
        y_final = res['x'].reshape((y_init.shape))
        # Rotate &  whiten to obtain canonical representation
        pca = PCA(whiten=True)
        self.coordinates = pca.fit_transform(y_final)
        #return self.coordinates
        
    def forward_mapping(self, x_new, k=3, c=0.01):
        """ Find intrinsic coordinates of x_new using k NNs
            k: no. of NNs, c: regularization """
        return self.mapping(self.x, self.coordinates,x_new,k,c)
        
    def backward_mapping(self, y_new, k=3, c=0.01):
        """ Find original coordinates of x_new using k NNs
            k: no. of NNs, c: regularization """
        return self.mapping(self.coordinates, self.x, y_new,k,c)
        
    ## UTILS : fit_tree ##
    def fit_ppca(self, x):
        """ x: neural activity (N,T) 
            returns: pca object
            Fit probabilistic PCA to samples x.
            Currently just fits diag-covariance Gaussian, no
            dimensionality reduction
            """
        pca = PCA(random_state=self.rng)
        pca.fit(x)
        pca.ll = pca.score(x)
        return pca
    
    def generate_hyperplanes(self, n_dir):
        """ Generate n_dir random splitting directions """
        hplanes = {}        
        v = self.rng.normal(0,1, size=(n_dir, self.dim))
        v /= np.sum(v**2, 1, keepdims=True)**0.5 # map to sphere
        hplanes['directions'] =  v
        hplanes['thresholds'] = self.rng.uniform(-1,1,size=n_dir)
        return hplanes
        
    def make_leaf(self, x, number,level):
        """ construct a leaf with x as samples. TO DO: idx"""
        pca = self.fit_ppca(x)
        leaf = {'type': 'leaf', 'level': level, 
                'number': number, 'samples': x, 
                'left': None, 'right': None, 'pca':pca}
        return leaf
        
    ## Utils: compute_distances ##
    def find_node(self, x0, start_node=None):
        """ Find the node of x_0, starting from start_node"""
        if start_node is None:
            # start recursion at root node 
            start_node = self.tree
            
        if start_node['type'] == 'leaf':
            return start_node
        else:
            v,c = start_node['v'], start_node['c']
            if v.dot(x0) < c:
                next_node = start_node['left']
            else:
                next_node = start_node['right']
        return self.find_node(x0, next_node)                             
        
    def get_prob(self, x_old, x_new):
        """ Get prob. of P(x_new|x_old) 
            under model of partition x_old """
        if self.tree is None:
            self.fit_tree()
        node = self.find_node(x_old)
        pca = node['pca']
        mu = pca.mean_
        C = pca.get_covariance() 
        C *= np.eye(self.dim) #extract diag. ?
        probs = multivariate_normal.pdf(x_new,mean=mu, cov=C)
        return probs/np.sum(probs)
        
    ## utils: learn coordinates ##
    def compute_probs(self):
        """ Compute transition probabilities """
        T = self.x.shape[0]        
        P = np.zeros((T,T))
        for i in range(T):
            P[i] = self.get_prob(self.x[i], self.x)
        self.P = P
    
    def compute_distances(self):
        """ Compute distances from probabilities
            TO DO: fully vectorize """
        if self.P is None:
            self.compute_probs()
        n = self.x.shape[0]
        eps = 1e-100
        D = np.sqrt(-np.log(self.P)+eps)
        #self.D_local = (D+D.T)/2
        D = shortest_path(D,method='J') # global distances
        self.D = (D + D.T)/2 # symmetrize
        #return self.D
        
    def loss_fun(self, y):
        """ Loss function for intrinsic coordinates y
            y: (n, dim) """
        if self.D is None:
            self.compute_distances
        eps = 1e-100
        D = self.D
        n = D.shape[0]
        y = y.reshape((n,self.manifold_dim))
        dif = np.sqrt(np.sum((y[:,np.newaxis] - y[np.newaxis,:])**2, axis=-1)+eps)
        error = 1/(D+np.eye(n)) * (D-dif)**2 # diag prevents 1/0, not counted in triu
        dist = np.sum(np.triu(error, k=1))
        return dist
        
    # Utils: fw/bw mapping 
    def mapping(self,s, t, s_new,k,c):
        """ Map s_new to t_new
            based on known mapping of s (source) to t (target),
            with s original/intrinsic coordinates
            and t intrinsic/original coordinates """
        n, s_dim = s.shape
        t_dim = t.shape[1]
        n_new = s_new.shape[0]
        # 1. determine nearest neighbors
        dist = np.sum((s[np.newaxis] - s_new[:,np.newaxis])**2,-1)
        nn_ids = np.argsort(dist)[:,:k] # change to [:,:k]
        nns = np.row_stack([s[nn_ids[:,ki]] for ki in range(k)])
        nns = nns.reshape((n_new, k, s_dim), order='F')
        # 2 determine gram matris; 
        dif = s_new[:,np.newaxis] - nns
        G = np.tensordot(dif,dif,axes=([2],[2]))
        G = G[np.arange(n_new),:,np.arange(n_new)]
        # 3. determine weights not worth vectorizing this 
        weights = np.zeros((n_new, k))
        for i_n in range(n_new): 
            weights[i_n] = np.linalg.inv(G[i_n]+c*np.eye(k)).dot(np.ones((k,)))
        weights /= np.sum(weights, -1, keepdims=True)
        # 4. compute coordinates
        t_nns = np.row_stack([t[nn_ids[:,ki]] for ki in range(k)])
        t_nns = t_nns.reshape((n_new,k, t_dim), order='F')
        t_new = np.dot(weights, t_nns)
        t_new = t_new[np.arange(n_new), np.arange(n_new)]
        return t_new
        
        
        
            
        

