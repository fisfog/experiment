#-*- coding:utf-8 -*-
'''
	author: Yunlei Mu
	email: muyunlei@gmail.com
'''

import numpy as np
import scipy as sp
import math
import time

def rand_mat(x, y):
	"""
	Generate a x*y matrix where the sum of every row is 1
	"""
	mat=[]
	for i in xrange(x):
		rand=[]
		mat.append([])
		for k in xrange(y):
			rand.append(np.random.random())
		norm=sum(rand)
		for j in xrange(y):
			mat[i].append(rand[j]*1.0/norm)
	return np.array(mat)

class myllda():
	"""
	LDA 

	--K: NUM of topic
	--M: NUM of document
	--V: NUM of words in vocabulary

	--doc: word index array

	--n_m_k: doc-topic count
	--n_k_t: topic-word count
	--n_m: SUM of each row in n_m_k
	--n_k: SUM of each row in n_k_t
	--phi: Parameters for topic-word distribution (topic*V)
	--theta: Parameters for doc-topic distribution (M*topic)
	--z: topic label array
	--alpha: doc-topic dirichlet prior parameter
	--beta: topic-word dirichlet prior parameter

	--max_iter: times of iterations

	"""
	def __init__(self,topic_num=5,it=50,ss=10,bs=40,result_path='./data/ldaResult/'):
		self.K = topic_num
		self.iterations = it
		self.saveStep = ss
		self.beginSaveIters = bs
		self.result_path = result_path


	def initializeModel(self, data):
		print "Initialize Model"
		self.M = data.doc_num
		self.V = data.v_num
		self.n_m_k = np.zeros((self.M, self.K))
		self.n_k_t = np.zeros((self.K, self.V))
		self.n_m = np.zeros(self.M)
		self.n_k = np.zeros(self.K)
		self.phi = rand_mat(self.K,self.V)
		self.theta = rand_mat(self.M,self.K)
		self.alpha = 50.0/self.K
		self.beta = 0.1
		# initialize documents index array
		self.doc = []
		for m in xrange(self.M):
			self.doc.append([])
			pnlist = data.id_doc_dict[data.docID[m]]
			N = len(pnlist)
			for n in xrange(N):
				self.doc[m].append(data.dictionary[pnlist[n]])

		# initialize topic label z for each word
		self.z = []
		for m in xrange(self.M):
			N = len(data.id_doc_dict[data.docID[m]])
			self.z.append([])
			for n in xrange(N):
				initTopic = np.random.randint(self.K)
				self.z[m].append(initTopic)
				self.n_m_k[m][initTopic] += 1
				self.n_k_t[initTopic][self.doc[m][n]] +=1
				self.n_k[initTopic] += 1
			self.n_m[m] = N

	def _update_parameters(self):
		"""
		Update multinomial parameters
		"""
		# update phi
		for k in xrange(self.K):
			s = self.n_k[k]+self.V*self.beta
			for t in xrange(self.V):
				self.phi[k][t] = (self.n_k_t[k][t]+self.beta)/s
		# update theta
		for m in xrange(self.M):
			s = self.n_m[m]+self.K*self.alpha
			for k in xrange(self.K):
				self.theta[m][k] = (self.n_m_k[m][k]+self.alpha)/s

	def _gibbs_sample(self,m,n):
		# Sample from p(z_i|z_-i,w) using Gibbs update rule

		# Remove topic label for w_{m,n}
		oldtopic = self.z[m][n]
		if self.n_m_k[m][oldtopic] > 0:
			self.n_m_k[m][oldtopic] -= 1
		if self.n_k_t[oldtopic][self.doc[m][n]] > 0:
			self.n_k_t[oldtopic][self.doc[m][n]] -= 1
		self.n_m[m] -= 1
		self.n_k[oldtopic] -= 1

		# Compute p(z_i=k|z_-i,w)
		pzi = np.zeros(self.K)
		for k in xrange(self.K):
			A = (self.n_k_t[k][self.doc[m][n]]+self.beta)/(self.n_k[k]+self.V*self.beta)
			B = (self.n_m_k[m][k]+self.alpha)/(self.n_m[m]+self.K*self.alpha)
			pzi[k] = A*B

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

		# Add new topic label for w_{m,n}
		self.n_m_k[m][newtopic] += 1
		self.n_k_t[newtopic][self.doc[m][n]] += 1
		self.n_m[m] += 1
		self.n_k[newtopic] += 1
		return newtopic

	def saveIteratedModel(self,iters):
		# lda.params,lda.phi,lda.theta,lda.tassign,lda.twords
		# lda.params
		res_Path = self.result_path
		model_name = "lda_"+str(iters)
		lines = []
		lines.append('alpha = '+ str(self.alpha))
		lines.append('beta = '+ str(self.beta))
		lines.append('topicNum = '+ str(self.K))
		lines.append('docNum = '+ str(self.M))
		lines.append('termNum = '+ str(self.V))
		lines.append('iterations = '+ str(self.iterations))
		lines.append('saveStep = '+ str(self.saveStep))
		lines.append('beginSaveIters = '+ str(self.beginSaveIters))
		f = open(res_Path+model_name+'.params','w')
		for s in lines:
			f.write(s+'\n')
		f.close()

		# lda.phi K*V
		f = open(res_Path+model_name+'.phi','w')
		for i in xrange(self.K):
			for j in xrange(self.V):
				f.write(str(self.phi[i][j])+'\t') 
			f.write('\n')
		f.close()

		# lda.theta M*K
		f = open(res_Path+model_name+'.theta','w')
		for i in xrange(self.M):
			for j in xrange(self.K):
				f.write(str(self.theta[i][j])+'\t')
			f.write('\n')
		f.close()

		# lda.tassign
		f = open(res_Path+model_name+'.tassign','w')
		for m in xrange(self.M):
			N = len(self.doc[m])
			for n in xrange(N):
				f.write(str(self.doc[m][n])+':'+str(self.z[m][n])+'\t')
			f.write('\n')
		f.close()

		# lda.twords phi[][] K*V
		# f = open(res_Path+model_name+'.twords','w')
		# top_num = 20
		# for i in xrange(self.K):
		# 	t_word_list = []
		# 	for j in xrange(self.V):
		# 		t_word_list.append(j)


	def inferenceModel(self):
		start = time.time()
		for step in xrange(self.iterations):
			print "Iteration %d"%step
			if step >= self.beginSaveIters and ((step-self.beginSaveIters)%self.saveStep==0):
				print "Saving model as iteration %d ..."%step
				self._update_parameters()
				self.saveIteratedModel(step)

			# use Gibbs Sampling to update z
			for m in xrange(self.M):
				N = len(self.doc[m])
				for n in xrange(N):
					newtopic = self._gibbs_sample(m,n)
					self.z[m][n] = newtopic
		end = time.time()
		print "Time: %f"%(end-start)


