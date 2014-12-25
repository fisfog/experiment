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

def randmat(n, d):
	mat = []
	for i in xrange(n):
		mat.append(np.random.random(d))
	return np.array(mat)

def h2w(h):
	sp = h.split('/')
	a = int(sp[0])
	b = int(sp[1])
	return (a*1.0+1)/(b+2)


class WSVD():
	"""
	"""
	def __init__(self,alpha=0.005,beta0=0.5,beta1=0.02,sr=0.99,dim=5,iters=60):
		self.alpha = alpha
		self.beta0 = beta0
		self.beta1 = beta1
		self.beta2 = 1
		self.beta3 = 1
		self.slowrate = sr
		self.dim = dim
		self.max_iter = iters

	def initializeModel(self,data,ldatheta):
		self.ldatheta = ldatheta
		self.recordN = data.recordN
		self.N = data.N
		self.M = data.M
		# self.user = data.uid_dict.keys()
		# self.product = data.pid_dict.keys()
		self.row = np.array(map(lambda x:data.uid_dict[x],[l[1] for l in data.traindata]))
		self.col = np.array(map(lambda x:data.pid_dict[x],[l[0] for l in data.traindata]))
		self.rate = np.array([float(l[3]) for l in data.traindata])
		self.helpfulness = np.array([h2w(l[2]) for l in data.traindata])
		self.mean = self.rate.mean()
		# self.b_u = randmat(data.N,1).reshape(data.N)
		# self.b_i = randmat(data.M,1).reshape(data.M)

		# initialize bu and bi with residuals
		self.b_u = np.zeros(self.N)
		self.b_i = np.zeros(self.M)
		self.R_u = np.zeros(self.N)
		self.R_i = np.zeros(self.M)
		for k in xrange(self.recordN):
			i = self.col[k]
			self.b_i[i] += self.rate[k]-self.mean
			self.R_i[i] += 1
		self.b_i /= self.beta2+self.R_i
		for k in xrange(self.recordN):
			u= self.row[k]
			i= self.col[k]
			self.b_u[u] += self.rate[k]-self.mean-self.b_i[i]
			self.R_u[u] += 1
		self.b_u[u] += 1
		self.b_u /= self.beta3+self.R_u
		self.p_u = randmat(self.N,self.dim)
		self.w = [0]*self.M
		self.theta = [0]*self.M
		c1 = np.zeros(self.M)
		c2 = np.zeros(self.M)
		# for pid in data.p_count_dic:
		# 	rd = np.random.rand()
		# 	self.w.append([rd]*data.p_count_dic[pid])
		for k in xrange(self.recordN):
			index = self.col[k]
			c1[index] += self.helpfulness[k]
			rd = np.random.rand()
			if self.w[index] == 0:
				self.w[index] = []
				self.w[index].append(rd)
				self.theta[index] = []
				self.theta[index].append(ldatheta[k])
				c2[index] += rd
			else:
				self.w[index].append(rd)
				self.theta[index].append(ldatheta[k])
				c2[index] += rd
		for k in xrange(self.recordN):
			index = self.col[k]
			self.helpfulness[k] /= c1[index]
		for k in xrange(self.M):
			for i in xrange(len(self.w[k])):
				self.w[k][i] /= c2[k]
		# self.w = np.array(self.w)
		# self.theta = ldatheta
		self.q_i = randmat(self.M,self.dim)

	def pred(self,u,i):
		# x = 0
		# for j in xrange(len(self.w[i])):
		# 	x += self.w[i][j]*self.theta[i][j]
		return self.mean+self.b_u[u]+self.b_i[i]+np.dot(self.p_u[u],self.q_i[i])

	def train_sgd(self):
		print "SGD start"
		start = time.time()
		preRmse = 1e10
		nowRmse = 0.0
		for step in xrange(self.max_iter):
			rmse = 0
			for k in xrange(self.recordN):
				u = self.row[k]
				i = self.col[k]
				eui = self.rate[k]-self.pred(u,i)
				rmse += math.pow(eui,2)
				self.b_u[u] += self.alpha*(eui-self.beta1*self.b_u[u])
				self.b_i[i] += self.alpha*(eui-self.beta1*self.b_i[i])
				self.q_i = np.zeros((self.M,self.dim))
				for j in xrange(len(self.w[i])):
					self.w[i][j] += self.alpha*(eui*np.dot(self.p_u[u],self.theta[i][j])+self.beta0*(self.helpfulness[k]-self.w[i][j]))
					self.q_i[i] += self.w[i][j]*self.theta[i][j]
				self.p_u[u] += self.alpha*(eui*self.q_i[i]-self.beta1*self.p_u[u])
			nowRmse = math.sqrt(rmse*1.0/self.recordN)
			if nowRmse >= preRmse and abs(preRmse-nowRmse)<=1e-5 and step>=3:
				break
			else:
				preRmse = nowRmse
			print "%d\t%f"%(step,nowRmse)
			self.alpha *= self.slowrate
		end = time.time()
		print "time:%f"%(end-start)
