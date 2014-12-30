#-*- coding:utf-8 -*-
'''
	author: Yunlei Mu
	email: muyunlei@gmail.com
'''
import numpy as np

def gen_stochastic_vec(x, y):
	"""
	Generate a x*y matrix where the sum of every row is 1
	"""
	r = np.random.random((x,y))
	return np.array([x/sum(x) for x in r])

def h2w(h,l1,l2):
	sp = h.split('/')
	a = int(sp[0])
	b = int(sp[1])
	return (a*1.0+l1)/(b+l2)
