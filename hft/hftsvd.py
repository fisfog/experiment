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

class mylsvd():
	"""
	SVD

	Parameters:
	--alpha: learning rate
	--slowrate: slow the learning rate
	--beta: regularization parameter 
	--dim: latent factor number
	--max_iter: max iteration times

	--recordN: record number
	--N: NUM of user
	--M: NUM of product
	--row: user index array of every record
	--col: product index array of every record
	--rate: rate array of every record
	--mean: overall average rating
	--b_u: user bias array
	--b_i: product bias array
	--p_u: user latent factor array
	--q_i: product latent factor array

	"""

	def __init__(self,alpha=0.005,beta=0.02,sr=0.99,dim=5,iters=60):
		self.alpha =  alpha
		self.beta = beta
		self.beta2 = 1
		self.beta3 = 1
		self.slowrate = sr
		self.dim = dim
		self.max_iter = iters

	def initializeModel(self,data,ldatheta,flag=0):
		self.ldatheta = ldatheta
		self.flag = flag
		self.recordN = data.recordN
		self.N = data.N
		self.M = data.M
		# self.user = data.uid_dict.keys()
		# self.product = data.pid_dict.keys()
		self.row = np.array(map(lambda x:data.uid_dict[x],[l[1] for l in data.traindata]))
		self.col = np.array(map(lambda x:data.pid_dict[x],[l[0] for l in data.traindata]))
		self.rate = np.array([float(l[3]) for l in data.traindata])
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
		self.p_u = randmat(data.N,self.dim)
		self.q_i = self.ldatheta

	def pred(self,u,i):
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
				self.b_u[u] += self.alpha*(eui-self.beta*self.b_u[u])
				self.b_i[i] += self.alpha*(eui-self.beta*self.b_i[i])
				if self.flag==0:
					self.q_i[i] += self.alpha*(eui*self.p_u[u]-self.beta*self.q_i[i])
				self.p_u[u] += self.alpha*(eui*self.q_i[i]-self.beta*self.p_u[u])
			nowRmse = math.sqrt(rmse*1.0/self.recordN)
			if nowRmse >= preRmse and abs(preRmse-nowRmse)<=1e-5 and step>=3:
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
		for k in xrange(self.recordN):
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
		a = [0]*n
		b = [0]*m
		x1 = [0]*(n*d)
		x2 = [0]*(m*d)
		for k in xrange(self.recordN):
			u = self.row[k]
			i = self.col[k]
			eui = self.rate[k]-self.mean-y1[u]-y2[i]-np.dot(pu[u*d:(u+1)*d],qi[i*d:(i+1)*d])
			a[u] += -2*eui+2*self.beta*y1[u]
			b[i] += -2*eui+2*self.beta*y2[i]
			for l in xrange(d):
				x1[u*d+l] += -2*eui*qi[i*d+l]+2*self.beta*pu[u*d+l]
				x2[i*d+l] += -2*eui*pu[u*d+l]+2*self.beta*qi[i*d+l]
		return np.array(a+b+x1+x2)


	def LBFGS(self):
		n = self.N
		m = self.M
		d = self.dim
		bu = self.b_u.reshape(self.N).tolist()
		bi = self.b_i.reshape(self.M).tolist()
		pu = utils.randmat(n,d).reshape(self.N*self.dim).tolist()
		qi = utils.randmat(m,d).reshape(self.M*self.dim).tolist()
		start = time.time()

		output = optimize.fmin_l_bfgs_b(self._lossfun,bu+bi+pu+qi,fprime=self._fprime,maxiter=self.max_iter)
		re = output[0]
		print output[1]
		self.b_u=re[:n]
		self.b_i=re[n:n+m]
		self.p_u=re[n+m:n+m+n*d].reshape((n,d))
		self.q_i=re[n+m+n*d:].reshape((m,d))

		end = time.time()
		print "time:%f"%(end-start)





class RMSE():
	"""
	"""
	def __init__(self,model,data):
		self.model = model
		self.test = data.testdata
		self.test_recordN = len(self.test)
		# self.test_row = np.array(map(lambda x:data.uid_dict[x],[l[1] for l in self.test]))
		# self.test_col = np.array(map(lambda x:data.pid_dict[x],[l[0] for l in self.test]))
		# self.rate = np.array([float(l[3]) for l in self.test])
		self.test_row = []
		self.test_col = []
		self.rate = []
		self.count = 0
		for k in xrange(self.test_recordN):
			l = self.test[k]
			if l[0] in data.pid_dict and l[1] in data.uid_dict:
				self.test_row.append(data.uid_dict[l[1]])
				self.test_col.append(data.pid_dict[l[0]])
				self.rate.append(float(l[3]))
				self.count += 1

	def compute(self):
		rmse = 0
		for k in xrange(self.count):
			u = self.test_row[k]
			i = self.test_col[k]
			eui = self.rate[k]-self.model.pred(u,i)
			rmse += math.pow(eui,2)
		return math.sqrt(rmse*1.0/self.test_recordN)