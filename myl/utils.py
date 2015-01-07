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

def avg_theta(ldamodel,metadata):
	"""
	avg 
	"""
	new_theta = np.zeros((metadata.M,ldamodel.K))
	# pidlist = metadata.pid_dict.keys()
	count = np.zeros(metadata.M)
	for i in xrange(metadata.recordN):
		pid = metadata.data[i][0]
		pindex = metadata.pid_dict[pid]
		count[pindex] += 1
		new_theta[pindex,:] += ldamodel.theta[i,:]
	for i in xrange(len(count)):
		new_theta[i,:] /= count[i]
	return new_theta

def hweighted_theta(ldamodel,metadata):
	"""
	use helpfulness rating weight theta
	"""
	new_theta = np.zeros((metadata.M,ldamodel.K))
	c = np.zeros(metadata.M)
	for i in xrange(metadata.recordN):
		pid = metadata.data[i][0]
		pindex = metadata.pid_dict[pid]
		w = h2w(metadata.data[i][2],1,2)
		c[pindex] += w
		new_theta[pindex,:] += w*ldamodel.theta[i,:]
	for i in xrange(len(c)):
		new_theta[i,:] /= c[i]
	return new_theta

def flatten(seq):
	for item in seq:
		for k in item:
			yield k

def result2file(res_dict,dataname):
	f = open('result.txt','a')
	f.write(dataname+":\n")
	for i in res_dict:
		f.write(i+':'+str(res_dict[i])+'\n')
	f.close()
