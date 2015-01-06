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

class HSVD():
	def __init__(self,dim=5,alpha=0.005,beta0=0.5,beta1=0.02,sr=0.99,iters=60,l1=1,l2=2):
		self.alpha =  alpha
		self.beta0 = beta0
		self.beta1 = beta1
		self.beta2 = 20
		self.beta3 = 20
		self.slowrate = sr
		self.dim = dim
		self.max_iter = iters
		self.l1 = l1
		self.l2 = l2

	def initialize_model(self,metadata,ldatheta):
		self.ldatheta = ldatheta
		self.total_record = metadata.recordN
		self.N = metadata.N
		self.M = metadata.M 
		self.row = np.array(map(lambda x:metadata.uid_dict[x],[l[1] for l in metadata.data]))
		self.col = np.array(map(lambda x:metadata.pid_dict[x],[l[0] for l in metadata.data]))
		self.rate = np.array([float(l[3]) for l in metadata.data])
		self.helpfulness = np.array([h2w(l[2],self.l1,self.l2) for l in metadata.data])
		self.train_record = int(self.total_record*0.8)
		self.val_record = int(self.total_record*0.1)
		self.test_record = self.total_record-self.train_record-self.val_record
		self.mean = self.rate[:self.train_record].mean()
		# estimate bu, bi
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

		self.p_u = np.random.random((self.N,self.dim))
		

		self.w = [[] for i in xrange(self.M)]
		self.h = [[] for i in xrange(self.M)]
		self.theta = [[] for i in xrange(self.M)]
		c1 = np.zeros(self.M)
		c2 = np.zeros(self.M)
		for k in xrange(self.total_record):
			p = self.col[k]
			rd = np.random.rand()
			self.w[p].append(rd)
			self.h[p].append(self.helpfulness[k])
			self.theta[p].append(self.ldatheta[k])
			c1[p] += rd
			c2[p] += self.helpfulness[k]
		for i in xrange(self.M):
			self.w[i] = [x/c1[i] for x in self.w[i]]
			self.h[i] = [x/c2[i] for x in self.h[i]]

		self.w = np.array(self.w)
		self.h = np.array(self.h)

		self.count = np.array([len(item) for item in self.w])
		self.acc_count = self.count.copy()
		for i in xrange(1,self.M):
			self.acc_count[i] += self.acc_count[i-1]

		# self._update_qi()
		self.q_i = np.random.random((self.M,self.dim))
		for k in xrange(self.train_record):
			i = self.col[k]
			self.q_i[i] = np.zeros(self.dim)
			for j in xrange(self.count[i]):
				self.q_i[i] += self.w[i][j]*self.theta[i][j]

	def _update_qi(self):
		self.q_i = np.zeros((self.M,self.dim))
		for i in xrange(self.M):
			for j in xrange(len(self.w[i])):
				self.q_i[i] += self.w[i][j]*self.theta[i][j]

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
				self.b_u[u] += self.alpha*(eui-self.beta1*self.b_u[u])
				self.b_i[i] += self.alpha*(eui-self.beta1*self.b_i[i])
				self.q_i[i] = np.zeros(self.dim)
				for j in xrange(self.count[i]):
					self.w[i][j] += self.alpha*(eui*np.dot(self.p_u[u],self.theta[i][j])+\
						self.beta0*(self.h[i][j]-self.w[i][j]))
					self.q_i[i] += self.w[i][j]*self.theta[i][j]
				self.p_u[u] += self.alpha*(eui*self.q_i[i]-self.beta1*self.p_u[u])
			nowRmse = math.sqrt(rmse*1.0/self.train_record)
			if nowRmse <= preRmse and abs(preRmse-nowRmse)<=1e-4 and step>=3:
				break
			else:
				preRmse = nowRmse
			print "%d\tRMSE:\t%f"%(step,nowRmse)
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
		w = x[n+m+n*d:]
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			if i==0:
				ind = 0
			else:
				ind = self.acc_count[i-1]
			qi = np.zeros(self.dim)
			wlf = 0
			for k in xrange(self.count[i]):
				qi += w[ind+k]*self.theta[i][k]
				wlf += (self.h[i][k]-w[ind+k])**2
			lf += (self.rate[k]-self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],qi))**2+wlf
		lf += self.beta1*(np.sum(y1**2)+np.sum(y2**2)+np.sum(pu**2))
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
		w = x[n+m+n*d:]
		a = [0 for i in xrange(n)]
		b = [0 for i in xrange(m)]
		x1 = [0 for i in xrange((n*d))]
		x2 = [0 for i in xrange(self.acc_count[-1])]
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			if i==0:
				ind = 0
			else:
				ind = self.acc_count[i-1]
			qi = np.zeros(self.dim)
			for k in xrange(self.count[i]):
				qi += w[ind+k]*self.theta[i][k]
			eui = self.rate[k]-self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],qi)
			a[u] += -2*eui+2*self.beta1*y1[u]
			b[i] += -2*eui+2*self.beta1*y2[i]
			for l in xrange(d):
				x1[u*d+l] += -2*eui*qi[l]+2*self.beta1*pu[u*d+l]
			for k in xrange(self.count[i]):
				x2[ind+k] += -2*eui*np.dot(pu[u*d:(u+1)*d],self.theta[i][k])\
					-2*self.beta0*(self.h[i][k]-w[ind+k])
		return np.array(a+b+x1+x2)

	def _cons(self):
		for k in xrange(self.M):
			def fun(x,i=k):
				w = x[self.N+self.M+self.N*self.dim:]
				if i==0:
					ind = 0
				else:
					ind = self.acc_count[i-1]
				s = sum(w[ind:ind+self.count[i]])
				return np.atleast_1d(1-s)
			yield fun

	def lbfgs(self):
		n = self.N
		m = self.M
		d = self.dim
		bu = self.b_u.reshape(self.N).tolist()
		bi = self.b_i.reshape(self.M).tolist()
		pu = self.p_u.reshape(self.N*self.dim).tolist()
		w = [x for x in flatten(self.w)]
		start = time.time()

		output = optimize.fmin_l_bfgs_b(self._lossfun,bu+bi+pu+w,fprime=self._fprime,maxiter=self.max_iter)

		re = output[0]
		print output[1]
		self.b_u=re[:n]
		self.b_i=re[n:n+m]
		self.p_u=re[n+m:n+m+n*d].reshape((n,d))
		tw = re[n+m+n*d:]
		for i in xrange(m):
			if i==0:
				ind = 0
			else:
				ind = self.acc_count[i-1]
			self.w[i] = tw[ind:ind+self.count[i]]
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

