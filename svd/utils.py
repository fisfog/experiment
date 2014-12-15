#-*- coding:utf-8 -*-

'''
	author: Yunlei Mu
	email: muyunlei@gmail.com

	file utils.py
'''

import numpy as np
import scipy as sp
import math
from scipy.sparse import bsr_matrix

def randmat(n, d):
	mat = []
	for i in xrange(n):
		mat.append(np.random.random(d))
	return np.array(mat)

class mylData():
	"""
	"""
	def __init__(self):
		self.sep = "::"
		self.uid = []
		self.iid = []
		self.rate = []
		self.time = []
		self.uid_dict = {}
		self.iid_dict = {}
		self.num_u = 0
		self.num_i = 0
		self.sparsity = 0

	def readdata(self,filename):
		print "Read data..."
		f=open(filename)
		for l in f.readlines():
			l=l.strip().split(self.sep)
			self.uid.append(l[0])
			self.iid.append(l[1])
			self.rate.append(float(l[2]))
			# self.tiem.append(l[3])
		f.close()
		k=0
		for ui in self.uid:
			if ui not in self.uid_dict:
				self.uid_dict[ui] = k
				k+=1
		k=0
		for iti in self.iid:
			if iti not in self.iid_dict:
				self.iid_dict[iti] = k
				k+=1
		self.num_u = len(self.uid_dict)
		self.num_i = len(self.iid_dict)
		print "%d users,%d items" %(self.num_u,self.num_i)
		self.row = np.array(map(lambda x:self.uid_dict[x],self.uid))
		self.col = np.array(map(lambda x:self.iid_dict[x],self.iid))
		self.rate = np.array(self.rate)
		self.M = len(self.rate)
		self.bsrmat = bsr_matrix((self.rate,(self.row,self.col)),shape=(self.num_u,self.num_i),dtype=float)
		self.sparsity = len(self.rate)*1.0/(self.num_u*self.num_i)
		print  "Mat Sparsity:%f"%self.sparsity

class numpyData():
	"""
	"""
	def __init__(self,sep,format):
		self.sep = sep
		self.format = format
		self.uid = []
		self.iid = []
		self.uid_dict = {}
		self.iid_dict = {}

	def readdata(self,filename):
		print "Read Data..."
		data = np.genfromtxt(filename,delimiter=self.sep,dtype=self.format['type'])
		self.record_no = len(data)
		print "Record No.:%d"%self.record_no
		for k in xrange(data.shape[1]):
			if k == self.format['row']:
				self.row = data[:,k]
			if k == self.format['col']:
				self.col = data[:,k]
			if k == self.format['rate']:
				self.rate = np.int32(data[:,k])
		p = 0
		q = 0
		for k in xrange(self.record_no):
			if self.row[k] not in self.uid_dict:
				self.uid_dict[self.row[k]] = p
				p += 1
			if self.col[k] not in self.iid_dict:
				self.iid_dict[self.col[k]] = q
				q += 1
		self.num_u = p
		self.num_i = q
		row = np.array(map(lambda x:self.uid_dict[x],self.row))
		col = np.array(map(lambda x:self.iid_dict[x],self.col))
		self.sparsity = self.record_no*100.0/(self.num_u*self.num_i)
		print  "Mat Sparsity:%f %%"%self.sparsity
		self.bsrmat = bsr_matrix((self.rate,(self.row,self.col)),shape=(self.num_u,self.num_i),dtype=float)
		

class RMSE():
	"""
	RMSE 
	"""
	def __init__(self,model,data):
		self.model = model
		self.data = data 		

	def compute(self):
		rmse = 0
		for k in xrange(self.data.M):
			u = self.data.row[k]
			i = self.data.col[k]
			eui = self.data.rate[k]-self.model.pred(u,i)
			rmse += math.pow(eui,2)
		return math.sqrt(rmse*1.0/self.data.M)
