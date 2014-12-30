# testscript

import parse
import hftlda
import hftsvd
import mylwsvd
import profile

def profileTest():
	filename = 'data/Arts.txt.gz'
	path = filename.strip('.txt.gz')+'/'
	r = []

	metadata = parse.amazonData()
	metadata.readdata(filename)

	Corpus_p = parse.amaCorpus()
	Corpus_p.set_data(metadata,form='product')

	lda = hftlda.myllda(result_path=path)
	lda.initializeModel(Corpus_p)
	lda.inferenceModel()

	only_svd = hftsvd.mylsvd()
	qi = hftsvd.randmat(metadata.M,only_svd.dim)
	only_svd.initializeModel(metadata,qi,flag=0)
	only_svd.train_sgd()

	rmse = hftsvd.RMSE(only_svd,metadata)
	r.append(rmse.compute())
	print "SVD ONLY RMSE:%f"%r[0]

	lda_svd = hftsvd.mylsvd()
	lda_svd.initializeModel(metadata,lda.theta,flag=1)
	lda_svd.train_sgd()

	rmse = hftsvd.RMSE(lda_svd,metadata)
	r.append(rmse.compute())
	print "LDA->SVD:%f"%r[1]

	Corpus_r = parse.amaCorpus()
	Corpus_r.set_data(metadata,form='review')

	lda_review = hftlda.myllda(result_path=path)
	lda_review.initializeModel(Corpus_r)
	lda_review.inferenceModel()

	newtheta = hftlda.hweighted_theta(lda_review,metadata)
	hsvd = hftsvd.mylsvd()
	hsvd.initializeModel(metadata,newtheta,flag=1)
	hsvd.train_sgd()

	rmse = hftsvd.RMSE(hsvd,metadata)
	r.append(rmse.compute())
	print "HWeighted SVD RMSE:%f"%r[2]

	wsvd = mylwsvd.WSVD(beta0=0.02)
	wsvd.initializeModel(metadata,lda_review.theta)
	wsvd.train_sgd()

	rmse = hftsvd.RMSE(wsvd,metadata)
	r.append(rmse.compute())
	print "HLMF RMSE:%f"%r[3]

	f=open('result.txt','a')
	s='\t'.join([str(x) for x in r])
	f.write(filename.strip('.txt.gz')+'\t'+s+'\n')
	f.close()


if __name__=='__main__':
	profile.run("profileTest()")



'''

filename = 'data/Automotive.txt.gz'
path = filename.strip('.txt.gz')+'/'
metadata = parse.amazonData()
metadata.readdata(filename)
Corpus_p = parse.amaCorpus()
Corpus_p.set_data(metadata,form='product')
lda = hftlda.myllda(result_path=path)
lda.initializeModel(Corpus_p)
lda.inferenceModel()
only_svd = hftsvd.mylsvd()
qi = hftsvd.randmat(metadata.M,only_svd.dim)
only_svd.initializeModel(metadata,qi,flag=0)
only_svd.train_sgd()
rmse = hftsvd.RMSE(only_svd,metadata)
r1 = rmse.compute()
print "SVD ONLY RMSE:%f"%r1
lda_svd = hftsvd.mylsvd()
lda_svd.initializeModel(metadata,lda.theta,flag=1)
lda_svd.train_sgd()
rmse = hftsvd.RMSE(lda_svd,metadata)
r2 = rmse.compute()
print "LDA->SVD:%f"%r2
Corpus_r = parse.amaCorpus()
Corpus_r.set_data(metadata,form='review')
lda_review = hftlda.myllda(result_path=path)
lda_review.initializeModel(Corpus_r)
lda_review.inferenceModel()
newtheta = hftlda.hweighted_theta(lda_review,metadata)
hsvd = hftsvd.mylsvd()
hsvd.initializeModel(metadata,newtheta,flag=1)
hsvd.train_sgd()
rmse = hftsvd.RMSE(hsvd,metadata)
r3 = rmse.compute()
print "HWeighted SVD RMSE:%f"%r3
wsvd = mylwsvd.WSVD(beta0=0.1)
wsvd.initializeModel(metadata,lda_review.theta)
wsvd.train_sgd()
rmse = hftsvd.RMSE(wsvd,metadata)
r4 = rmse.compute()
print "HLMF RMSE:%f"%r4
f=open('result.txt','a')
f.write(filename.strip('.txt.gz')+str(r1)+str(r2)+str(r3)+str(r4)+'\n')
f.close()

'''