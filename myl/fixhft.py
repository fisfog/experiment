#-*- coding:utf-8 -*-
'''
	author: Yunlei Mu
	email: muyunlei@gmail.com
'''

import numpy as np
import scipy as sp
import math
import time
from itertools import izip
from scipy import optimize
from utils import *

class HFT():
	def __init__(self,dim=5,em_it=50,grad_it=50,lbd=0.1,latent_reg=0):
		self.beta2 = 1
		self.beta3 = 1
		self.K = dim
		self.em_iters = em_it
		self.grad_iters = grad_it
		self.lbd = lbd
		self.latent_reg = latent_reg

	def initialize_model(self,metadata):
		#					 #	
		#   read data        #
		#                    #
		self.total_record = metadata.recordN
		self.id2word_dict = dict(izip(metadata.vocab_dict.itervalues(),metadata.vocab_dict.iterkeys()))
		self.N = metadata.N
		self.M = metadata.M
		self.V = metadata.V
		self.row = np.array(map(lambda x:metadata.uid_dict[x],[l[1] for l in metadata.data]))
		self.col = np.array(map(lambda x:metadata.pid_dict[x],[l[0] for l in metadata.data]))
		self.rate = np.array([float(l[3]) for l in metadata.data])
		self.words = [l[-1] for l in metadata.data]
		self.train_record = int(self.total_record*0.8)
		self.val_record = int(self.total_record*0.1)
		self.test_record = self.total_record-self.train_record-self.val_record
		
		# total number of parameters
		self.NW = 1+1+(self.K+1)*(self.N+self.M)
		# Initialize parameters and latent variables
		# Zero all weights
		self.W = np.zeros(self.NW)
		self.mean,self.kappa,self.b_u,self.b_i,self.p_u,self.q_i = self._get_parameter(self.W)

		self.mean = self.rate[:self.train_record].mean()
		error = self.valid_test_error()
		print "Error offset term only (train/valid/test) = %f/%f/%f (%f)"%error

		# estimate the parameters is by decoupling the calculation of the bi’s from the calculation of the bu’s

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

		error = self.valid_test_error()
		print "Error offset and bias (train/valid/test) = %f/%f/%f (%f)"%error

		if self.lbd > 0:
			self.mean = 0
			self.b_u = np.zeros(self.N)
			self.b_i = np.zeros(self.M)


		self.n_m_k = np.zeros((self.M, self.K))
		self.n_k_t = np.zeros((self.K, self.V))
		self.n_m = np.zeros(self.M)
		self.n_k = np.zeros(self.K)
		self.z = [[] for i in xrange(self.train_record)]
		# initialize topic label z for each word
		for k in xrange(self.train_record):
			i = self.col[k]
			w = self.words[k]
			self.n_m[i] += len(w)
			for wp in w:
				t = np.random.randint(self.K)
				self.z[k].append(t)
				self.n_m_k[i][t] += 1
				self.n_k_t[t][wp] += 1
				self.n_k[t] += 1

		if self.lbd == 0:
			self.p_u = np.random.random((self.N,self.K))
			self.q_i = np.random.random((self.M,self.K))

		self.kappa = 1.0
		self.phi = gen_stochastic_vec(self.K,self.V)
		if self.lbd > 0:
			self._update_topics(True)

	def _get_W(self,mean,kappa,b_u,b_i,p_u,q_i):
		w = np.zeros(self.NW)
		w[0] = mean
		w[1] = kappa
		w[2:2+self.N] = b_u
		w[2+self.N:2+self.N+self.M] = b_i
		w[2+self.N+self.M:2+self.N+self.M+self.K*self.N] = p_u.reshape(self.K*self.N)
		w[2+self.N+self.M+self.K*self.N:2+self.N+self.M+self.K*self.N+self.K*self.M] = \
					q_i.reshape(self.K*self.M)
		return w

	def _get_parameter(self,x):
		mean = x[0]
		kappa = x[1]
		b_u = x[2:2+self.N]
		b_i = x[2+self.N:2+self.N+self.M] 
		p_u = x[2+self.N+self.M:2+self.N+self.M+self.K*self.N].reshape((self.N,self.K))
		q_i = x[2+self.N+self.M+self.K*self.N:2+self.N+self.M+self.K*self.N+self.K*self.M].reshape((self.M,self.K))
		return mean,kappa,b_u,b_i,p_u,q_i

	def valid_test_error(self):
		train = 0
		valid = 0
		test = 0
		test_ste = 0
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			train += (self.rate[k]-self.pred(u,i))**2
		for k in xrange(self.train_record,self.train_record+self.val_record):
			u = self.row[k]
			i = self.col[k]
			valid += (self.rate[k]-self.pred(u,i))**2
		for k in xrange(self.train_record+self.val_record,self.total_record):
			u = self.row[k]
			i = self.col[k]
			err = (self.rate[k]-self.pred(u,i))**2
			test += err
			test_ste += err*err
		train /= self.train_record
		valid /= self.val_record
		test /= self.test_record
		test_ste /= self.test_record
		test_ste = math.sqrt((test_ste-test*test)/self.test_record)
		return train,valid,test,test_ste

	def _topic_Z(self,i):
		'''
		Compute normalization constant for a particular item
		'''
		self.res = np.exp(self.kappa*self.q_i[i]).sum()

	def _update_topics(self,sample):
		'''
		Update topic assingments for each word.
		If sample==True, this is done by sampling, otherwise it's done by maximum likelihood
		'''
		for k in xrange(self.train_record):
			if k > 0 and k%100000 == 0:
				print '.',
			i = self.col[k]
			u = self.row[k]
			w = self.words[k]
			topics = self.z[k]
			self._topic_Z(i)
			for wp in xrange(len(w)):
				wi = w[wp]
				pw = np.zeros(self.K)
				A = np.exp(self.kappa*self.q_i[i])/self.res
				for t in xrange(self.K):
					B = (self.n_k_t[t][wi]+0.1)/(self.n_k[t]+self.V*0.1)
					pw[t] = A[t]*B
				newtopic = 0
				if sample:
					for i in xrange(1,self.K):
						pw[i] += pw[i-1]
					u = np.random.rand()*pw[-1]
					while newtopic < self.K:
						if u < pw[newtopic]:
							break
						newtopic += 1
				else:
					newtopic = np.argmax(pw)

				if newtopic != topics[wp]:
					t = topics[wp]
					self.n_k_t[t][wi] -= 1
					self.n_k_t[newtopic][wi] +=1
					self.n_k[t] -= 1
					self.n_k[newtopic] += 1
					self.n_m_k[i][t] -= 1
					self.n_m_k[i][newtopic] += 1
					self.z[k][wp] = newtopic

	def pred(self,u,i):
		return self.mean+self.b_u[u]+self.b_i[i]+np.dot(self.p_u[u],self.q_i[i])


	def _lsq(self,x):
		'''
		Compute the energy according to the least_squares criterion
		'''

		lsq_start = time.time()

		mean,kappa,b_u,b_i,p_u,q_i = self._get_parameter(x)
		res = 0
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			rate = self.rate[k]
			res += (mean+b_u[u]+b_i[i]+np.dot(p_u[u],q_i[i])-rate)**2

		for b in xrange(self.M):
			self._topic_Z(b)
			lZ = np.log(self.res)
			for k in xrange(self.K):
				res += -self.lbd*self.n_m_k[b][k]*(kappa*q_i[b][k]-lZ)

		if self.latent_reg > 0:
			res += self.latent_reg*(np.sum(p_u**2)+np.sum(q_i**2))

		for k in xrange(self.K):
			for w in xrange(self.V):
				res += -self.lbd*self.n_k_t[k][w]*np.log(self.phi[k][w])

		lsq_end = time.time()

		return res

	def _dl(self,x):
		'''
		Derivative of the energy function
		'''
		dl_start = time.time()

		mean,kappa,b_u,b_i,p_u,q_i = self._get_parameter(x)
		w = np.zeros(self.NW)
		dmean,dkappa,db_u,db_i,dp_u,dq_i = self._get_parameter(w)
		for k in xrange(self.train_record):
			u = self.row[k]
			i = self.col[k]
			rate = self.rate[k]
			dl = 2*(mean+b_u[u]+b_i[i]+np.dot(p_u[u],q_i[i])-rate)
			dmean += dl
			db_u[u] += dl
			db_i[i] += dl
			dp_u[u] += dl*q_i[i]
			dq_i[i] += dl*p_u[u]

		for i in xrange(self.M):
			self._topic_Z(i)
			qtZ = np.sum(q_i[i]*np.exp(kappa*q_i[i]))
			for k in xrange(self.K):
				dq_i[i] += -self.lbd*kappa*self.n_m_k[i][k]*(1-np.exp(kappa*q_i[i][k])/self.res)
				dkappa += -self.lbd*self.n_m_k[i][k]*(q_i[i][k]-qtZ/self.res)

		if self.latent_reg > 0:
			dp_u += self.latent_reg*2*p_u
			dq_i += self.latent_reg*2*q_i

		w = self._get_W(dmean,dkappa,db_u,db_i,dp_u,dq_i)
		return w

	def disp_top_words(self):
		print "Top wors for each topic:"
		for k in xrange(self.K):
			bestwordid = np.argsort(self.phi[k])[-10:]
			print "Topic%d:"%k,
			for w in bestwordid:
				print self.id2word_dict[w],
			print '\n'

	def _update_phi(self):
		"""
		Update multinomial parameters
		"""
		# update phi
		for k in xrange(self.K):
			s = self.n_k[k]+self.V*0.1
			for t in xrange(self.V):
				self.phi[k][t] = (self.n_k_t[k][t]+0.1)/s

	
	def train(self):
		# parameter training for HFT
		# mean,kappa,bu,bi,pu,qi,phi
		best_valid = 1e8
		for emi in xrange(self.em_iters):
			self.w = self._get_W(self.mean,self.kappa,self.b_u,self.b_i,self.p_u,self.q_i)
			output = optimize.fmin_l_bfgs_b(func=self._lsq,x0=self.w,fprime=self._dl,maxiter=self.grad_iters)
			print "energy after gradient setp = %f"%output[1]
			self.mean,self.kappa,self.b_u,self.b_i,self.p_u,self.q_i = self._get_parameter(output[0])

			if self.lbd>0:
				self._update_topics(True)
				self._update_phi()
				self.disp_top_words()

			error = self.valid_test_error()
			print "Error offset term only (train/valid/test) = %f/%f/%f (%f)"%error

			if error[1]<best_valid:
				best_valid = error[1]














