#-*- coding:utf-8 -*-

import numpy as np
import scipy as sp
from scipy import optimize


class test():
	def __init__(self):
		self.k = 5
		self.x = np.random.random(self.k)
		self.y = np.random.random(self.k)

	def _lsf(self,x):
		# return np.dot(self.x,self.y)
		a1 = x[:self.k]
		a2 = x[self.k:]
		return np.dot(a1,a2)

	def _df(self,x):
		a1 = x[:self.k].tolist()
		a2 = x[self.k:].tolist()
		return np.array(a2+a1)

	def lbfgs(self):
		x0 = self.x.tolist()+self.y.tolist()
		output = optimize.fmin_l_bfgs_b(self._lsf,x0,fprime=self._df)
		print output