#-*- coding:utf-8 -*-
'''
	author: Yunlei Mu
	email: muyunlei@gmail.com
'''

import numpy as np
import scipy as sp
import math
import time
import utils
from scipy import optimize

class svd():
	"""
	"""
	def __init__(self,alpha=0.005,beta=0.02,sr=0.99,dim=10):
		self.alpha =  alpha
		self.beta = beta
		self.beta2 = 1
		self.beta3 = 1
		self.slowrate = sr
		self.dim = dim

	def set_data(self,data):
		self.data = data
		self.mean = data.rate.mean()
		self.b_u = utils.randmat(data.num_u,1).reshape(data.num_u)
		self.b_i = utils.randmat(data.num_i,1).reshape(data.num_i)
		self.p_u = utils.randmat(data.num_u,self.dim)
		self.q_i = utils.randmat(data.num_i,self.dim)

	def pred(self,u,i):
		return self.mean+self.b_u[u]+self.b_i[i]+np.dot(self.p_u[u],self.q_i[i])
		# return np.dot(self.p_u[u],self.q_i[i])


	def SGDtrain(self,max_iter=60):
		print "SGD Start..."
		start = time.clock()
		preRmse = 1e10
		nowRmse = 0.0
		for step in xrange(max_iter):
			rmse = 0
			for k in xrange(self.data.M):
				u = self.data.row[k]
				i = self.data.col[k]
				eui = self.data.rate[k]-self.pred(u,i)
				rmse += math.pow(eui,2)
				self.b_u[u] += self.alpha*(eui-self.beta*self.b_u[u])
				self.b_i[i] += self.alpha*(eui-self.beta*self.b_i[i])
				self.q_i[i] += self.alpha*(eui*self.p_u[u]-self.beta*self.q_i[i])
				self.p_u[u] += self.alpha*(eui*self.q_i[i]-self.beta*self.p_u[u])
			nowRmse = math.sqrt(rmse*1.0/self.data.M)
			if nowRmse >= preRmse and abs(preRmse-nowRmse)<=1e-5 and step>=3:
				break
			else:
				preRmse = nowRmse
			print "%d\t%f"%(step,nowRmse)
			self.alpha *= self.slowrate
		end = time.clock()
		print "time:%f"%(end-start)

	def estimate_b(self):
		self.eb_u = np.zeros(self.data.num_u)
		self.eb_i = np.zeros(self.data.num_i)
		self.R_u = np.zeros(self.data.num_u)
		self.R_i = np.zeros(self.data.num_i)
		for k in xrange(self.data.M):
			i = self.data.col[k]
			self.eb_i[i] += self.data.rate[k] - self.mean
			self.R_i[i] += 1
		self.eb_i /= self.beta2+self.R_i
		for k in xrange(self.data.M):
			u = self.data.row[k]
			i = self.data.col[k]
			self.eb_u[u] += self.data.rate[k] - self.mean - self.eb_i[i]
			self.R_u[u] += 1
		self.eb_u /= self.beta3+self.R_u
		rmse = 0
		for k in xrange(self.data.M):
			u = self.data.row[k]
			i = self.data.col[k]
			eui = self.data.rate[k] - self.mean-self.eb_u[u]-self.eb_i[i]
			rmse += math.pow(eui,2)
		rmse = math.sqrt(rmse*1.0/self.data.M)
		print "RMSE:%f"%rmse

		

	def _lossfun(self,x):
		lf = 0
		n = self.data.num_u
		m = self.data.num_i
		d = self.dim
		y1 = x[:n]
		y2 = x[n:n+m]
		pu = x[n+m:n+m+n*d]
		qi = x[n+m+n*d:]
		for k in xrange(self.data.M):
			u = self.data.row[k]
			i = self.data.col[k]
			lf += (self.data.rate[k] - self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],qi[i*d:(i+1)*d]))**2
		lf += self.beta*(np.sum(y1**2)+np.sum(y2**2)+np.sum(pu**2)+np.sum(qi**2))
		return lf

	def _fprime(self,x):
		"""
		Gradient
		"""
		n = self.data.num_u
		m = self.data.num_i
		d = self.dim
		y1 = x[:n]
		y2 = x[n:n+m]
		pu = x[n+m:n+m+n*d]
		qi = x[n+m+n*d:]
		a = [0]*n
		b = [0]*m
		x1 = [0]*(n*d)
		x2 = [0]*(m*d)
		for k in xrange(self.data.M):
			u = self.data.row[k]
			i = self.data.col[k]
			eui = self.data.rate[k] - self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],qi[i*d:(i+1)*d])
			a[u] += -2*eui+2*self.beta*y1[u]
			b[i] += -2*eui+2*self.beta*y2[i]
			for l in xrange(d):
				x1[u*d+l] += -2*eui*qi[i*d+l]+2*self.beta*pu[u*d+l]
				x2[i*d+l] += -2*eui*pu[u*d+l]+2*self.beta*qi[i*d+l]
		return np.array(a+b+x1+x2)


	def LBFGS(self,max_iter=100):
		n = self.data.num_u
		m = self.data.num_i
		d = self.dim
		bu = utils.randmat(self.data.num_u,1).reshape(self.data.num_u).tolist()
		bi = utils.randmat(self.data.num_i,1).reshape(self.data.num_i).tolist()
		pu = utils.randmat(n,d).reshape(self.data.num_u*self.dim).tolist()
		qi = utils.randmat(m,d).reshape(self.data.num_i*self.dim).tolist()
		start = time.time()

		re=optimize.fmin_l_bfgs_b(self._lossfun,bu+bi+pu+qi,fprime=self._fprime,maxiter=max_iter)[0]
		self.b_u=re[:n]
		self.b_i=re[n:n+m]
		self.p_u=re[n+m:n+m+n*d].reshape((n,d))
		self.q_i=re[n+m+n*d:].reshape((m,d))

		end = time.time()
		print "time:%f"%(end-start)


########################unittest############################### 

import unittest

class TestSVD(unittest.TestCase):
	def test_svd(self):
		train = utils.mylData()
		# train.sep='\t'
		train.readdata('data/yelp_rating.dat')
		train_svd = svd(alpha=0.01,beta=0.05)
		train_svd.set_data(train)
		train_svd.SGDtrain(150)

if __name__=='__main__':
	unittest.main()

