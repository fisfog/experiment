#-*- coding:utf-8 -*-
'''
	author: Yunlei Mu
	email: muyunlei@gmail.com
'''

import numpy as np
import scipy as sp
import math
import time
from scipy import optimize
from utils import *

class HFT():
	def __init__(self,mu=0.1,dim=5,it=5):
		self.mu = mu
		self.beta2 = 20
		self.beta3 = 20
		self.K = dim
		self.iter_sum = it
		self.max_iter = 30

	def initialize_model(self,metadata,corpus):
		#					 #	
		#   read data        #
		#                    #
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

		#					 #	
		# init svd parameter #
		#                    #
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

		#					 #	
		# init lda parameter #
		#                    #
		self.V = corpus.v_num
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
		self.kappa = 1.0

	def pred(self,u,i):
		return self.mean+self.b_u[u]+self.b_i[i]+np.dot(self.p_u[u],self.q_i[i])

	def _update_theta(self):
		for i in xrange(self.M):
			s = np.sum(np.exp(self.kappa*self.q_i[i]))
			self.theta[i] = np.exp(self.kappa*self.q_i[i])/s

	def _lossfun(self,x):
		rat_err = 0
		likelihood = 0
		n = self.N
		m = self.M
		d = self.K
		v = self.V
		bu = x[:n]
		bi = x[n:n+m]
		pu = np.array(x[n+m:n+m+n*d]).reshape((n,d))
		qi = np.array(x[n+m+n*d:n+m+n*d+m*d]).reshape((m,d))
		phi = np.array(x[n+m+n*d+m*d:n+m+n*d+m*d+d*v]).reshape((d,v))
		kp = x[-1]
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			rat_err += (self.rate[k]-bu[u]-bi[i]-np.dot(pu[u],qi[i]))**2
		theta = np.zeros((m,d))
		for i in xrange(m):
			s = np.sum(np.exp(kp*qi[i]))
			theta[i] = np.exp(kp*qi[i])/s
		for i in xrange(m):
			N = len(self.doc[i])
			for j in xrange(N):
				likelihood += np.log(theta[i][self.z[i][j]])+np.log(phi[self.z[i][j]][self.doc[i][j]])
		likelihood *= self.mu
		return rat_err-likelihood

	def _fprime(self,x):
		n = self.N
		m = self.M
		d = self.K
		v = self.V
		bu = x[:n]
		bi = x[n:n+m]
		pu = np.array(x[n+m:n+m+n*d]).reshape((n,d))
		qi = np.array(x[n+m+n*d:n+m+n*d+m*d]).reshape((m,d))
		phi = np.array(x[n+m+n*d+m*d:n+m+n*d+m*d+d*v]).reshape((d,v))
		kp = x[-1]
		bu_p = np.zeros(n)
		bi_p = np.zeros(m)
		pu_p = np.zeros((n,d))
		qi_p = np.zeros((m,d))
		phi_p = np.zeros((d,v))
		kp_p = 0
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-bu[u]-bi[i]-np.dot(pu[u],qi[i])
			bu_p[u] -= 2*eui
			bi_p[i] -= 2*eui
			pu_p[u] -= 2*eui*qi[i]
			qi_p[i] -= 2*eui*pu[u]
		theta = np.zeros((m,d))
		for i in xrange(m):
			s = np.sum(np.exp(kp*qi[i]))
			theta[i] = np.exp(kp*qi[i])/s
		for i in xrange(m):
			N = len(self.doc[i])
			for k in xrange(d):
				s0 = qi[i][k]*np.exp(kp*qi[i][k])
			for j in xrange(N):
				s = np.sum(np.exp(kp*qi[i]))
				qi_p[i][self.z[i][j]] -= kp*(1-np.exp(kp*qi[i][self.z[i][j]])/s)
				phi_p[self.z[i][j]][self.doc[i][j]] -= self.mu/phi[self.z[i][j]][self.doc[i][j]]
				kp_p -= qi[i][self.z[i][j]]-s0/s
		bu_p = bu_p.tolist()
		bi_p = bi_p.tolist()
		pu_p = pu_p.reshape(n*d).tolist()
		qi_p = qi_p.reshape(m*d).tolist()
		phi_p = phi_p.reshape(d*v).tolist()
		kp_p = [kp_p]
		return np.array(bu_p+bi_p+pu_p+qi_p+phi_p+kp_p)

	def _lbgfs(self):
		n = self.N
		m = self.M
		d = self.K
		v = self.V
		bu = self.b_u.tolist()
		bi = self.b_i.tolist()
		pu = self.p_u.reshape(n*d).tolist()
		qi = self.q_i.reshape(m*d).tolist()
		phi = self.phi.reshape(d*v).tolist()
		kp = [self.kappa]
		start = time.time()
		output = optimize.fmin_l_bfgs_b(self._lossfun,bu+bi+pu+qi+phi+kp,fprime=self._fprime,maxiter=self.max_iter)
		
		print "loss:%f"%output[1]
		re = output[0]
		self.b_u = np.array(re[:n])
		self.b_i = np.array(re[n:n+m])
		self.p_u = np.array(re[n+m:n+m+n*d]).reshape((n,d))
		self.q_i = np.array(re[n+m+n*d:n+m+n*d+m*d]).reshape((m,d))
		self.phi = np.array(re[n+m+n*d+m*d:n+m+n*d+m*d+d*v]).reshape((d,v))
		self.kappa = re[-1]

		end = time.time()
		print "time:%f"%(end-start)

	def _updata_varphi(self):
		# make phi as a sochastic vector
		self.varphi = np.zeros((self.K,self.V))
		for k in xrange(self.K):
			s = np.sum(np.exp(self.phi[k]))
			self.varphi[k] = self.phi[k]/s


	def _sampler(self,d,j):
		oldtopic = self.z[d][j]
		pzi = np.zeros(self.K)
		for k in xrange(self.K):
			pzi[k] = self.varphi[k][self.doc[d][j]]

		# Sample a new topic label for w_{m,n}
		# Compute cumulated probability for pzi
		for k in xrange(1,self.K):
			pzi[k] += pzi[k-1]

		u = np.random.rand()*pzi[-1]
		newtopic = 0
		while newtopic < self.K:
			if u < pzi[newtopic]:
				break
			newtopic += 1

		return newtopic

	def cal_train_mse(self):
		mse = 0
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-self.pred(u,i)
			mse += math.pow(eui,2)
		return math.sqrt(mse*1.0/self.train_record)

	def train(self):
		# parameter training for HFT
		# bu,bi,pu,qi,phi,theta*,kappa
		print "Begin Training, %d iteraters for one step"%self.max_iter

		for step in xrange(self.iter_sum):
			print "Step%d:"%step
			print "update parameter using LBFGS"
			self._lbgfs()
			self._updata_varphi()
			self._update_theta()
			print "MSE:%f"%self.cal_train_mse()
			print "Sample z_dj"			
			for d in xrange(self.M):
				N = len(self.doc[d])
				for j in xrange(N):
					newtopic = self._sampler(d,j)
					self.z[d][j] = newtopic

	def cal_val_rmse(self):
		return math.sqrt(self.cal_val_mse())

	def cal_val_mse(self):
		mse = 0
		for k in xrange(self.train_record,self.train_record+self.val_record):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-self.pred(u,i)
			mse += math.pow(eui,2)
		return math.sqrt(mse*1.0/self.val_record)	

	def cal_test_rmse(self):
		return math.sqrt(self.cal_test_mse())

	def cal_test_mse(self):
		mse = 0
		for k in xrange(self.train_record+self.val_record,self.total_record):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-self.pred(u,i)
			mse += math.pow(eui,2)
		return mse*1.0/self.test_record













