#-*- coding:utf-8 -*-
'''
	author: Yunlei Mu
	email: muyunlei@gmail.com
'''

import gzip
# import simplejson
import numpy as np
import scipy as sp
import re
import nltk
import enchant
import time

def parse(filename):
	"""
	Parse Amazon Dataset
	"""
	f = gzip.open(filename, 'r')
	entry = {}
	for l in f:
		l = l.strip()
		colonPos = l.find(':')
		if colonPos == -1:
			yield entry
			entry = {}
			continue
		eName = l[:colonPos]
		rest = l[colonPos+2:]
		entry[eName] = rest
	yield entry

def load_stop_words(sw_file_path):
	"""
	Load StopWords list
	"""
	StopWords = set()
	sw_file = open(sw_file_path, "r")
	for word in sw_file:
		word = word.replace("\n", "")
		word = word.replace("\r\n", "")
		StopWords.add(word)
	sw_file.close()
	return StopWords

def h2w(h,l1,l2):
	sp = h.split('/')
	a = int(sp[0])
	b = int(sp[1])
	return (a*1.0+l1)/(b+l2)

class amadata_proc(object):
	PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']
	CARRIAGE_RETURNS = ['\n', '\r\n']
	WORD_REGEX = "^[a-z']+$"
	def __init__(self,filename,sw_file_path = '../data/stopwords.txt'):
		start = time.time()
		self.StopWords = load_stop_words(sw_file_path)
		print "Reading Data..."
		self.data = [[e['product/productId'],e['review/userId'],\
					e['review/helpfulness'],e['review/score'],e['review/time'],e['review/text']] \
					for e in parse(filename) if e!={}]
		self.recordN = len(self.data)
		print "Total Record num:%d"%self.recordN
		print "Review Text Preprocessing"
		self.simple_words = [self._simple_proc([x[-1]]) for x in self.data]
		self.lda_words = [self._text_preproc([x[-1]]) for x in self.data]
		end = time.time()
		print "Preprocessing Complete, time:%f"%(end-start)

		dlt = []
		for i in xrange(len(self.lda_words)):
			if len(self.simple_words)==0:
				dlt.append(i)
			if len(self.lda_words[i])==0:
				self.lda_words[i] = self.simple_words[i]

		for i in dlt:
			del self.data[i]
			del self.simple_words[i]
			del self.lda_words[i]
			self.recordN -= 1

		# code users and items
		self.uid_dict = {}
		self.pid_dict = {}
		ucount = 0 
		pcount = 0 
		for k in xrange(self.recordN):
			pname = self.data[k][0]
			uname = self.data[k][1]
			if pname not in self.pid_dict:
				self.pid_dict[pname] = pcount
				pcount += 1
			if uname not in self.uid_dict:
				self.uid_dict[uname] = ucount
				ucount += 1
		self.N = len(self.uid_dict)
		self.M = len(self.pid_dict)
		print "%d users\t%d products"%(self.N,self.M)
		self.sparsity = self.recordN*1.0/(self.N*self.M)
		print "Mat Sparsity:%f%%"%(self.sparsity*100)



	def write2file(self,outfile):
		f = open(outfile,'w')
		for k in xrange(self.recordN):
			f.write(self.data[k][1]+' '+self.data[k][0]+' '+self.data[k][2]+' '+self.data[k][3]+' '+\
				self.data[k][4]+' '+str(len(self.simple_words[k]))+' ')
			for w in self.simple_words[k]:
				if w != self.simple_words[k][-1]:
					f.write(w+' ')
				else:
					f.write(w)
			f.write('\n')
		f.close()

	def write2lda_doc(self,outfile,form):
		f = open(outfile,'w')
		if form == 'review':
			f.write(str(self.recordN)+'\n')
			for k in xrange(self.recordN):
				for w in self.lda_words[k]:
					if w != self.lda_words[k][-1]:
						f.write(w+' ')
					else:
						f.write(w)
				f.write('\n')
		if form == 'item':
			words_for_item = [[] for i in xrange(self.M)]
			for k in xrange(self.recordN):
				pindex = self.pid_dict[self.data[k][0]]
				words_for_item[pindex] += self.lda_words[k]
			f.write(str(self.M)+'\n')
			for wl in words_for_item:
				for w in wl:
					if w != wl[-1]:
						f.write(w+' ')
					else:
						f.write(w)
				f.write('\n')
		if form == 'pnitem':
			words_for_item = [[] for i in xrange(self.M)]

		f.close()



	def _clean_word(self,word):
		"""
		Convert words to lowercase
		Del the PUNCTUATION and CARRIAGE_RETURNS
		"""
		word = word.lower()
		for punc in self.PUNCTUATION+self.CARRIAGE_RETURNS:
			word = word.replace(punc,"").strip("'")
		return word

	def _simple_proc(self,textlist):
		text = []
		for line in textlist:
			words =  nltk.word_tokenize(line)
			# words = line.split(' ')
			for w in words:
				clean_word = self._clean_word(w)
				if clean_word:
					text.append(clean_word)
		return text

	def _text_preproc(self,textlist):
		"""
		Text Preprocessing
		Include:
		1.Clean word
		2.Participle
		3.Spell checking
		4.Stemming
		"""
		text = []
		ck = enchant.Dict("en_US")
		porter = nltk.PorterStemmer()
		for line in textlist:
			words =  nltk.word_tokenize(line)
			# words = line.split(' ')
			for w in words:
				clean_word = self._clean_word(w)
				if clean_word and ck.check(clean_word) and \
				clean_word not in self.StopWords and len(clean_word)>1:
					text.append(porter.stem(clean_word))
		return text