#-*- coding:utf-8 -*-

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

class amazonData(object):
	"""
	the class described Amazon Dataset
	Attribute:

	--traindata&testdata: a list of DataItem
	[0]: product/productId
	[1]: review/userId
	[2]: review/helpfulness
	[3]: review/score
	[4]: review/text

	--uid_dict: key:userId value:userIndex

	--pid_dict: key:productId value:productIndex

	--recordN: SUM of Record

	--N: SUM of Users

	--M: SUM of Product

	--sparsity: Sparsity of user-product matrix
	"""
	
	def __init__(self):
		self.traindata = []
		self.testdata = []
		
	def readdata(self,filename):
		print "Read Data..."
		totaldata = []
		for e in parse(filename):
			if e=={}:
				break
			if e['review/userId']!='unknown':
				totaldata.append([e['product/productId'],e['review/userId'],e['review/helpfulness'],e['review/score'],e['review/text']])

			# t = np.random.rand()
			# if t<0.8:
			# 	self.traindata.append([e['product/productId'],e['review/userId'],e['review/helpfulness'],e['review/score'],e['review/text']])
			# else:
			# 	self.testdata.append([e['product/productId'],e['review/userId'],e['review/helpfulness'],e['review/score'],e['review/text']])
		print "Total Record num:%d"%len(totaldata)
		# split train/test data
		
		self.p_count_dic = {}
		self.u_count_dic = {}
		for item in totaldata:
			if item[0] not in self.p_count_dic:
				self.p_count_dic[item[0]] = 1
			else:
				self.p_count_dic[item[0]] += 1
			if item[1] not in self.u_count_dic:
				self.u_count_dic[item[1]] = 1
			else:
				self.u_count_dic[item[1]] += 1
		
		# 1
		self.ucand = [x for x in self.u_count_dic if self.u_count_dic[x] == 1]
		self.pcand = [x for x in self.p_count_dic if self.p_count_dic[x] == 1]
		for l in totaldata:
			if l[1] not in self.ucand:
				r = np.random.rand()
				if r < 0.8:
					self.traindata.append(l)
				else:
					self.testdata.append(l)

		'''
		# 2
		testindex = []
		pcount = 0
		for pid in self.p_count_dic:
			c = self.p_count_dic[pid]
			c /= 3
			for i in xrange(c):
				rd = np.random.randint(3)
				testindex.append(pcount+rd)
			pcount += self.p_count_dic[pid]

		for i in xrange(len(totaldata)):
			if i in testindex:
				self.testdata.append(totaldata[i])
			else:
				self.traindata.append(totaldata[i])
		'''
		# process traindata 
		self.recordN = len(self.traindata)
		print "Record No.:%d"%self.recordN

		self.uid_dict = {}
		self.pid_dict = {}
		ucount = 0
		pcount = 0
		for i in xrange(self.recordN):
			pname = self.traindata[i][0]
			uname = self.traindata[i][1]
			if pname not in self.pid_dict:
				self.pid_dict[pname] = pcount
				pcount += 1
			if uname not in self.uid_dict:
				self.uid_dict[uname] = ucount
				ucount += 1
		self.N = len(self.uid_dict)
		self.M = len(self.pid_dict)
		print "%d user\t%d product"%(self.N,self.M)
		self.sparsity = self.recordN*1.0/(self.N*self.M)
		print "Mat Sparsity:%f%%"%(self.sparsity*100)

		c = 0
		for l in self.testdata:
			if l[0] in self.pid_dict and l[1] in self.uid_dict:
				c += 1
		print c,c*1.0/len(self.testdata)

		# process testdata

class amaCorpus():
	"""
	Amazon data review Corpus Class

	--StopWords: stopwords set from file

	--doc_num: SUM of document

	--v_num: SUM of Vocabulary

	--doc_form: way of document orgnization
	'product': all reviews of a product as a document
	'user': all reviews of a user as a document
	'review': one review as a document

	--id_doc_dict: dict: key:docId value:review words

	--vocab: Vocabulary list

	--docID: productId list

	"""
	PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*']
	CARRIAGE_RETURNS = ['\n', '\r\n']
	WORD_REGEX = "^[a-z']+$"
	def __init__(self,sw_file_path = './data/stopwords.txt'):
		self.doc_num = 0
		self.v_num = 0
		self.StopWords = load_stop_words(sw_file_path)


	def _clean_word(self,word):
		"""
		Convert words to lowercase
		Del the PUNCTUATION and CARRIAGE_RETURNS
		"""
		word = word.lower()
		for punc in amaCorpus.PUNCTUATION+amaCorpus.CARRIAGE_RETURNS:
			word = word.replace(punc,"").strip("'")
		return word if re.match(amaCorpus.WORD_REGEX,word) else None

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
				if clean_word and ck.check(clean_word) and clean_word not in self.StopWords and len(clean_word)>1:
					text.append(porter.stem(clean_word))
		return text

	def set_data(self,data,form):
		start = time.time()
		self.doc_form = form.lower()
		# Doc orginization way: product
		if self.doc_form == "product":
			self.doc_num = data.M
			self.id_doc_dict = {}
			for i in xrange(data.recordN):
				pname = data.traindata[i][0]
				if pname not in self.id_doc_dict:
					self.id_doc_dict[pname] = []
					self.id_doc_dict[pname].append(data.traindata[i][-1])
				else:
					self.id_doc_dict[pname].append(data.traindata[i][-1])
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
			self.doc_num = data.N
			self.id_doc_dict = {}
			for i in xrange(data.recordN):
				uname = data.traindata[i][1]
				if uname not in self.id_doc_dict:
					self.id_doc_dict[uname] = []
					self.id_doc_dict[uname].append(data.traindata[i][-1])
				else:
					self.id_doc_dict[uname].append(data.traindata[i][-1])
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
			self.doc_num = data.recordN
			self.id_doc_dict = {}
			for i in xrange(data.recordN):
				self.id_doc_dict[i] = []
				self.id_doc_dict[i].append(data.traindata[i][-1])
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


