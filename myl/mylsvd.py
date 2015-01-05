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

class SVD():
	def __init__(self,dim=5,alpha=0.005,beta=0.02,sr=0.99,iters=60):
		self.alpha =  alpha
		self.beta = beta
		self.beta2 = 20
		self.beta3 = 20
		self.slowrate = sr
		self.dim = dim
		self.max_iter = iters

	def initialize_model(self,metadata,ldatheta,flag=True):
		self.ldatheta = ldatheta
		self.flag = flag
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
		self.b_u = np.zeros(self.N)
		self.b_i = np.zeros(self.M)
		self.R_u = np.zeros(self.N)
		self.R_i = np.zeros(self.M)
		for k in xrange(self.train_record):
			i = self.col[k]
			self.b_i[i] += self.rate[k]-self.mean
			self.R_i[i] += 1
		self.b_i /= self.beta2+self.R_i
		for k in xrange(self.train_record):
			u= self.row[k]
			i= self.col[k]
			self.b_u[u] += self.rate[k]-self.mean-self.b_i[i]
			self.R_u[u] += 1
		self.b_u[u] += 1
		self.b_u /= self.beta3+self.R_u

		# self.b_u = np.random.random(self.N)
		# self.b_i = np.random.random(self.M)

		self.p_u = np.random.random((self.N,self.dim))
		self.q_i = np.random.random((self.M,self.dim))

	def pred(self,u,i):
		return self.mean+self.b_u[u]+self.b_i[i]+np.dot(self.p_u[u],self.q_i[i])

	def sgd(self):
		print "SGD start"
		start = time.time()
		preRmse = 1e10
		nowRmse = 0.0
		for step in xrange(self.max_iter):
			rmse = 0
			for k in xrange(self.train_record):
				u = self.row[k]
				i = self.col[k]
				eui = self.rate[k]-self.pred(u,i)
				rmse += math.pow(eui,2)
				self.b_u[u] += self.alpha*(eui-self.beta*self.b_u[u])
				self.b_i[i] += self.alpha*(eui-self.beta*self.b_i[i])
				if self.flag:
					self.q_i[i] += self.alpha*(eui*self.p_u[u]-self.beta*self.q_i[i])
				else:
					self.q_i[i] = self.ldatheta[i]
				self.p_u[u] += self.alpha*(eui*self.q_i[i]-self.beta*self.p_u[u])
			nowRmse = math.sqrt(rmse*1.0/self.train_record)
			if nowRmse >= preRmse and abs(preRmse-nowRmse)<=1e-4 and step>=3:
				break
			else:
				preRmse = nowRmse
			print "%d\t%f"%(step,nowRmse)
			self.alpha *= self.slowrate
		end = time.time()
		print "time:%f"%(end-start)

	def _lossfun(self,x):
		lf = 0
		n = self.N
		m = self.M
		d = self.dim
		y1 = x[:n]
		y2 = x[n:n+m]
		pu = x[n+m:n+m+n*d]
		qi = x[n+m+n*d:]
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			lf += (self.rate[k]-self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],qi[i*d:(i+1)*d]))**2
		lf += self.beta*(np.sum(y1**2)+np.sum(y2**2)+np.sum(pu**2)+np.sum(qi**2))
		return lf

	def _fprime(self,x):
		"""
		Gradient
		"""
		n = self.N
		m = self.M
		d = self.dim
		y1 = x[:n]
		y2 = x[n:n+m]
		pu = x[n+m:n+m+n*d]
		qi = x[n+m+n*d:]
		a = [0 for i in xrange(n)]
		b = [0 for i in xrange(m)]
		x1 = [0 for i in xrange((n*d))]
		x2 = [0 for i in xrange((m*d))]
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],qi[i*d:(i+1)*d])
			a[u] += -2*eui+2*self.beta*y1[u]
			b[i] += -2*eui+2*self.beta*y2[i]
			for l in xrange(d):
				x1[u*d+l] += -2*eui*qi[i*d+l]+2*self.beta*pu[u*d+l]
				x2[i*d+l] += -2*eui*pu[u*d+l]+2*self.beta*qi[i*d+l]
		return np.array(a+b+x1+x2)

	def _lossfun_l(self,x):
		lf = 0
		n = self.N
		m = self.M
		d = self.dim
		y1 = x[:n]
		y2 = x[n:n+m]
		pu = x[n+m:n+m+n*d]
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			lf += (self.rate[k]-self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],self.q_i[i]))**2
		lf += self.beta*(np.sum(y1**2)+np.sum(y2**2)+np.sum(pu**2))
		return lf

	def _fprime_l(self,x):
		"""
		Gradient
		"""
		n = self.N
		m = self.M
		d = self.dim
		y1 = x[:n]
		y2 = x[n:n+m]
		pu = x[n+m:n+m+n*d]
		a = [0 for i in xrange(n)]
		b = [0 for i in xrange(m)]
		x1 = [0 for i in xrange((n*d))]
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],self.q_i[i])
			a[u] += -2*eui+2*self.beta*y1[u]
			b[i] += -2*eui+2*self.beta*y2[i]
			for l in xrange(d):
				x1[u*d+l] += -2*eui*self.q_i[i][l]+2*self.beta*pu[u*d+l]
		return np.array(a+b+x1)

	def LBFGS(self):
		n = self.N
		m = self.M
		d = self.dim
		bu = self.b_u.reshape(self.N).tolist()
		bi = self.b_i.reshape(self.M).tolist()
		pu = np.random.random((n,d)).reshape(self.N*self.dim).tolist()
		qi = np.random.random((m,d)).reshape(self.M*self.dim).tolist()
		start = time.time()
		if self.flag:
			output = optimize.fmin_l_bfgs_b(self._lossfun,bu+bi+pu+qi,fprime=self._fprime,maxiter=self.max_iter)
		else:
			output = optimize.fmin_l_bfgs_b(self._lossfun_l,bu+bi+pu,fprime=self._fprime_l,maxiter=self.max_iter)
		re = output[0]
		print output[1]
		self.b_u=re[:n]
		self.b_i=re[n:n+m]
		self.p_u=re[n+m:n+m+n*d].reshape((n,d))
		if self.flag:
			self.q_i=re[n+m+n*d:].reshape((m,d))
		end = time.time()
		print "time:%f"%(end-start)

	def cal_val_rmse(self):
		return math.sqrt(self.cal_val_mse())

	def cal_val_mse(self):
		rmse = 0
		for k in xrange(self.train_record,self.train_record+self.val_record):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-self.pred(u,i)
			rmse += math.pow(eui,2)
		return math.sqrt(rmse*1.0/self.val_record)	

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

	def offset_mse(self):
		mse = 0
		for k in xrange(self.train_record+self.val_record,self.total_record):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-self.mean
			mse += math.pow(eui,2)
		return mse*1.0/self.test_record

