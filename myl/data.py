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
					for e in parse(filename) if e!={} and e['review/userId']!='unknown']
		self.recordN = len(self.data)
		print "Total Record num:%d"%self.recordN
		print "Review Text Preprocessing"
		self.words = [self._text_preproc([x[-1]]) for x in self.data]
		end = time.time()
		print "Preprocessing Complete, time:%f"%(end-start)

	def write2file(self,outfile):
		f = open(outfile,'w')
		for k in xrange(self.recordN):
			f.write(self.data[k][0]+'\t'+self.data[k][1]+'\t'+self.data[k][2]+'\t'+self.data[k][3]+'\t'+\
				self.data[k][4]+'\t')
			for w in self.words[k]:
				f.write(w+',')
			f.write('\n')
		f.close()

	def _clean_word(self,word):
		"""
		Convert words to lowercase
		Del the PUNCTUATION and CARRIAGE_RETURNS
		"""
		word = word.lower()
		for punc in amacorpus.PUNCTUATION+amacorpus.CARRIAGE_RETURNS:
			word = word.replace(punc,"").strip("'")
		return word if re.match(amacorpus.WORD_REGEX,word) else None

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


class amameta():
	def __init__(self,datafile):
		print "Read data"
		f = open(datafile)
		self.data = [l.strip('\n').split('\t') for l in f]
		f.close()
		self.recordN = len(self.data)
		for k in xrange(self.recordN):
			self.data[k][-1] = self.data[k][-1].strip(',').split(',')
		# shuffle the data
		np.random.shuffle(self.data)

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

		# build vocabulary dict
		self.vocab_dict = {}
		wcount = 0
		for i in xrange(self.recordN):
			for w in self.data[i][-1]:
				if w not in self.vocab_dict:
					self.vocab_dict[w] = wcount
					wcount += 1
		self.V = len(self.vocab_dict)
		print "Total %d words"%self.V

		for k in xrange(self.recordN):
			self.data[k][-1] = [self.vocab_dict[x] for x in self.data[k][-1]]


class amacorpus():
	PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']
	CARRIAGE_RETURNS = ['\n', '\r\n']
	WORD_REGEX = "^[a-z']+$"
	def __init__(self,sw_file_path = '../data/stopwords.txt'):
		self.StopWords = load_stop_words(sw_file_path)

	def _clean_word(self,word):
		"""
		Convert words to lowercase
		Del the PUNCTUATION and CARRIAGE_RETURNS
		"""
		word = word.lower()
		for punc in amacorpus.PUNCTUATION+amacorpus.CARRIAGE_RETURNS:
			word = word.replace(punc,"").strip("'")
		return word if re.match(amacorpus.WORD_REGEX,word) else None

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

	def data_proc(self,metadata,form):
		start = time.time()
		self.doc_form = form.lower()
		if self.doc_form == "product":
			self.doc_num = metadata.M
			self.id_doc_dict = {}
			for i in xrange(metadata.recordN):
				pname = metadata.data[i][0]
				if pname not in self.id_doc_dict:
					self.id_doc_dict[pname] = []
					self.id_doc_dict[pname].append(metadata.data[i][-1])
				else:
					self.id_doc_dict[pname].append(metadata.data[i][-1])
			print "Review Text Preprocessing"
			for pd in self.id_doc_dict:
				rw = self.id_doc_dict[pd]
				self.id_doc_dict[pd] = self._text_preproc(rw)
			print "Preprocessing Complete"

			self.docID = self.id_doc_dict.keys()

			print "Build Word Dictionary"
			self.dictionary = {}
			count = 0
			for pd in self.id_doc_dict:
				rw = self.id_doc_dict[pd]
				for w in rw:
					if w not in self.dictionary:
						self.dictionary[w] = count
						count += 1
			self.v_num = len(self.dictionary)
			print "Total %d words in Corpus"%self.v_num
			end = time.time()
			print "Total time:%f"%(end-start)

		# Doc orginization way: review 
		if self.doc_form == "user":
			self.doc_num = metadata.N
			self.id_doc_dict = {}
			for i in xrange(metadata.recordN):
				uname = metadata.data[i][1]
				if uname not in self.id_doc_dict:
					self.id_doc_dict[uname] = []
					self.id_doc_dict[uname].append(metadata.data[i][-1])
				else:
					self.id_doc_dict[uname].append(metadata.data[i][-1])
			print "Review Text Preprocessing"
			for pd in self.id_doc_dict:
				rw = self.id_doc_dict[pd]
				self.id_doc_dict[pd] = self._text_preproc(rw)
			print "Preprocessing Complete"

			self.docID = self.id_doc_dict.keys()

			print "Build Word Dictionary"
			self.dictionary = {}
			count = 0
			for pd in self.id_doc_dict:
				rw = self.id_doc_dict[pd]
				for w in rw:
					if w not in self.dictionary:
						self.dictionary[w] = count
						count += 1
			self.v_num = len(self.dictionary)
			print "Total %d words in Corpus"%self.v_num
			end = time.time()
			print "Total time:%f"%(end-start)

		# Doc orginization way: review 
		if self.doc_form == "review":
			self.doc_num = metadata.recordN
			self.id_doc_dict = {}
			for i in xrange(metadata.recordN):
				self.id_doc_dict[i] = []
				self.id_doc_dict[i].append(metadata.data[i][-1])
			print "Review Text Preprocessing"
			for pd in self.id_doc_dict:
				rw = self.id_doc_dict[pd]
				self.id_doc_dict[pd] = self._text_preproc(rw)
			print "Preprocessing Complete"
			self.docID = self.id_doc_dict.keys()
			print "Build Word Dictionary"
			self.dictionary = {}
			count = 0
			for pd in self.id_doc_dict:
				rw = self.id_doc_dict[pd]
				for w in rw:
					if w not in self.dictionary:
						self.dictionary[w] = count
						count += 1
			self.v_num = len(self.dictionary)
			print "Total %d words in Corpus"%self.v_num
			end = time.time()
			print "Total time:%f"%(end-start)

	def text_proc(self,meta):
		print "Review Text Preprocessing"
		self.words = [[] for i in xrange(meta.recordN)]
		for k in xrange(meta.recordN):
			self.words[k] = self._text_preproc([meta.data[k][-1]])
		print "Preprocessing Complete"


	def write2file(self,filename):
		f = open(filename,'w')
		f.write(self.doc_form+'\n')
		for item in self.id_doc_dict:
			f.write(str(item)+'\t')
			for w in self.id_doc_dict[item]:
				f.write(w+',')
			f.write('\n')
		f.close()

	def read_form_file(self,filename):
		self.id_doc_dict = {}
		f = open(filename)
		self.doc_form = f.readline().strip('\n')
		for l in f:
			l = l.strip('\n').split('\t')
			self.id_doc_dict[l[0]] = l[1].split(',')
			self.id_doc_dict[l[0]].pop()
		f.close()
		self.docID = self.id_doc_dict.keys()
		self.doc_num = len(self.id_doc_dict)
		print "Build Word Dictionary"
		self.dictionary = {}
		count = 0
		for pd in self.id_doc_dict:
			rw = self.id_doc_dict[pd]
			for w in rw:
				if w not in self.dictionary:
					self.dictionary[w] = count
					count += 1
		self.v_num = len(self.dictionary)
		print "Total %d words in Dictionary"%self.v_num

		



