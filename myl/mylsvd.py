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
		self.beta2 = 1
		self.beta3 = 1
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
		self.q_i = self.ldatheta

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

