#-*- coding:utf-8 -*-
'''
	author: Yunlei Mu
	email: muyunlei@gmail.com
'''

import numpy as np
import scipy as sp
import math
import time
# from scipy import optimize
from utils import *

class HFT():
	def __init__(self,mu=0.1,dim=5):
		self.mu = mu
		self.beta2 = 20
		self.beta3 = 20
		self.K = dim

	def initialize_model(self,metadata,corpus):
		self.total_record = metadata.recordN
		self.N = metadata.N
		self.M = metadata.M 
		self.row = np.array(map(lambda x:metadata.uid_dict[x],[l[1] for l in metadata.data]))
		self.col = np.array(map(lambda x:metadata.pid_dict[x],[l[0] for l in metadata.data]))
		self.rate = np.array([float(l[3]) for l in metadata.data])
		self.train_record = int(self.total_record*0.8)
		self.val_record = int(self.total_record*0.1)
		self.test_record = self.total_record-self.train_record-self.val_record
		self.mean = self.rate[:self.train_record].mean()

		# init svd parameter
		self.b_u = np.zeros(self.N)
		self.b_i = np.zeros(self.M)
		self.R_u = np.zeros(self.N)
		self.R_i = np.zeros(self.M)
		for k in xrange(self.total_record):
			i = self.col[k]
			self.b_i[i] += self.rate[k]-self.mean
			self.R_i[i] += 1
		self.b_i /= self.beta2+self.R_i
		for k in xrange(self.total_record):
			u= self.row[k]
			i= self.col[k]
			self.b_u[u] += self.rate[k]-self.mean-self.b_i[i]
			self.R_u[u] += 1
		self.b_u[u] += 1
		self.b_u /= self.beta3+self.R_u
		self.p_u = np.random.random((self.N,self.K))
		self.q_i = np.random.random((self.M,self.K))

		# init lda parameter
		self.V = corpus.v_num
		self.n_m_k = np.zeros((self.M, self.K))
		self.n_k_t = np.zeros((self.K, self.V))
		self.n_m = np.zeros(self.M)
		self.n_k = np.zeros(self.K)
		self.phi = gen_stochastic_vec(self.K,self.V)
		self.theta = gen_stochastic_vec(self.M,self.K)
		self.alpha = 50.0/self.K
		self.beta = 0.1
		# initialize documents index array
		self.doc = []
		for m in xrange(self.M):
			self.doc.append([])
			pnlist = corpus.id_doc_dict[corpus.docID[m]]
			N = len(pnlist)
			for n in xrange(N):
				self.doc[m].append(corpus.dictionary[pnlist[n]])

		# initialize topic label z for each word
		self.z = []
		for m in xrange(self.M):
			N = len(corpus.id_doc_dict[corpus.docID[m]])
			self.z.append([])
			for n in xrange(N):
				initTopic = np.random.randint(self.K)
				self.z[m].append(initTopic)
				self.n_m_k[m][initTopic] += 1
				self.n_k_t[initTopic][self.doc[m][n]] +=1
				self.n_k[initTopic] += 1
			self.n_m[m] = N

	def pred(self,u,i):
		return self.mean+self.b_u[u]+self.b_i[i]+np.dot(self.p_u[u],self.q_i[i])

	def _qi2theta(self,i):
		s = np.sum(np.exp(self.kappa*self.q_i[i]))
		self.theta[i] = np.exp(self.kappa*self.q_i[i])/s

	def 


