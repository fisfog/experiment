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
# from scipy.sparse import bsr_matrix

class baseline():
	"""
	The baseline predictor

	b_ui = \mu+b_u+b_i
	"""
	def __init__(self, alpha=0.01, beta=0.05):
		self.mean = 0
		self.alpha = alpha
		self.beta1 = beta
		self.beta2 = 1
		self.beta3 = 1
		self.slowrate = 0.99
	
	def set_data(self,data):
		self.data = data
		self.mean = data.rate.mean()
		self.b_u = utils.randmat(self.data.num_u,1).reshape(self.data.num_u)
		self.b_i = utils.randmat(self.data.num_i,1).reshape(self.data.num_i)

	def reset(self):
		self.b_u = utils.randmat(self.data.num_u,1).reshape(self.data.num_u)
		self.b_i = utils.randmat(self.data.num_i,1).reshape(self.data.num_i)



	def pred(self, u,i):
		return self.mean+self.b_u[u]+self.b_i[i]

	def SGDtrain(self,max_iter=60):
		self.b_u = utils.randmat(self.data.num_u,1).reshape(self.data.num_u)
		self.b_i = utils.randmat(self.data.num_i,1).reshape(self.data.num_i)
		# self.b_u = np.zeros(self.data.num_u)
		# self.b_i = np.zeros(self.data.num_i)
		print "SGD Start..."
		start = time.clock()
		preRmse = 1e10
		nowRmse = 0.0
		for step in xrange(max_iter):
			rmse = 0
			# a=0
			# b=0
			# c=0
			for k in xrange(self.data.M):

				u = self.data.row[k]
				i = self.data.col[k]

				# eui = self.data.rate[k]-self.pred(u,i)
				eui = self.data.rate[k] - self.mean-self.b_u[u]-self.b_i[i]

				# a += time.clock()

				rmse += math.pow(eui,2)

				# b += time.clock()

				self.b_u[u] += self.alpha*(eui-self.beta1*self.b_u[u])
				self.b_i[i] += self.alpha*(eui-self.beta1*self.b_i[i])

				# c += time.clock()
			# print b-a,c-b,c-a
			nowRmse = math.sqrt(rmse*1.0/self.data.M)
			if nowRmse >= preRmse and abs(preRmse-nowRmse)<=1e-5 and step>=3:
				break
			else:
				preRmse = nowRmse
			print "%d\t%f"%(step,nowRmse)
			self.alpha *= self.slowrate
		print "Interation Complete!"
		end = time.clock()
		print "time:%f"%(end-start)

	def estimate(self):
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
		y1 = x[:self.data.num_u]
		y2 = x[self.data.num_u:]
		for k in xrange(self.data.M):
			u = self.data.row[k]
			i = self.data.col[k]
			lf += (self.data.rate[k] - self.mean-y1[u]-y2[i])**2
		lf += self.beta1*(np.sum(y1**2)+np.sum(y2**2))
		return lf

	def _fprime(self,x):
		y1 = x[:self.data.num_u]
		y2 = x[self.data.num_u:]
		a = [0]*self.data.num_u
		b = [0]*self.data.num_i
		for k in xrange(self.data.M):
			u = self.data.row[k]
			i = self.data.col[k]
			eui = self.data.rate[k] - self.mean-y1[u]-y2[i]
			a[u] += -2*eui+2*self.beta1*y1[u]
			b[i] += -2*eui+2*self.beta1*y2[i]
		return np.array(a+b)


	def LBFGS(self,max_iter=45):
		b_u = utils.randmat(self.data.num_u,1).reshape(self.data.num_u).tolist()
		b_i = utils.randmat(self.data.num_i,1).reshape(self.data.num_i).tolist()
		start = time.clock()

		re=optimize.fmin_l_bfgs_b(self._lossfun,b_u+b_i,fprime=self._fprime,maxiter=max_iter)[0]
		self.b_u=re[:self.data.num_u]
		self.b_i=re[self.data.num_u:]

		end = time.clock()
		print "time:%f"%(end-start)



########################unittest############################### 

import unittest

class TestSVD(unittest.TestCase):
	def test_svd(self):
		train = utils.mylData()
		# train.sep='\t'
		train.readdata('data/yelp_rating.dat')
		train_baseline = baseline(alpha=0.01,beta=0.05)
		train_baseline.set_data(train)
		train_baseline.SGDtrain(150)

if __name__=='__main__':
	unittest.main()

