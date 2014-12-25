# testscript

import parse
import hftlda
import hftsvd
import mylwsvd
import profile

def profileTest():
	filename = 'data/Shoes.txt.gz'
	path = filename.strip('.txt.gz')+'/'

	metadata = parse.amazonData()
	metadata.readdata(filename)

	Corpus_p = parse.amaCorpus()
	Corpus_p.set_data(metadata,form='product')

	lda = hftlda.myllda(result_path=path+'product/')
	lda.initializeModel(Corpus_p)
	lda.inferenceModel()

	only_svd = hftsvd.mylsvd()
	qi = hftsvd.randmat(metadata.M,only_svd.dim)
	only_svd.initializeModel(metadata,qi,flag=0)
	only_svd.train_sgd()

	rmse = hftsvd.RMSE(only_svd,metadata)
	print "SVD ONLY RMSE:%f"%rmse.compute()

	lda_svd = hftsvd.mylsvd()
	lda_svd.initializeModel(metadata,lda.theta,flag=1)
	lda_svd.train_sgd()

	rmse = hftsvd.RMSE(lda_svd,metadata)
	print "LDA->SVD:%f"%rmse.compute()

	Corpus_r = parse.amaCorpus()
	Corpus_r.set_data(metadata,form='review')

	lda_review = hftlda.myllda(result_path=path+'review/')
	lda_review.initializeModel(Corpus_r)
	lda_review.inferenceModel()

	newtheta = hftlda.hweighted_theta(lda_review,metadata)
	hsvd = hftsvd.mylsvd()
	hsvd.initializeModel(metadata,newtheta,flag=1)
	hsvd.train_sgd()

	rmse = hftsvd.RMSE(hsvd,metadata)
	print "HWeighted SVD RMSE:%f"%rmse.compute()

	wsvd = mylwsvd.WSVD()
	wsvd.initializeModel(metadata,lda_review.theta)
	wsvd.train_sgd()

	rmse = hftsvd.RMSE(wsvd,metadata)
	print "HLMF RMSE:%f"%rmse.compute()


if __name__=='__main__':
	profile.run("profileTest()")



'''
filename = 'data/Shoes.txt.gz'
path = filename.strip('.txt.gz')+'/'
metadata = parse.amazonData()
metadata.readdata(filename)
Corpus_p = parse.amaCorpus()
Corpus_p.set_data(metadata,form='product')
lda = hftlda.myllda(result_path=path+'product/')
lda.initializeModel(Corpus_p)
lda.inferenceModel()
only_svd = hftsvd.mylsvd()
qi = hftsvd.randmat(metadata.M,only_svd.dim)
only_svd.initializeModel(metadata,qi,flag=0)
only_svd.train_sgd()
rmse = hftsvd.RMSE(only_svd,metadata)
print "SVD ONLY RMSE:%f"%rmse.compute()
lda_svd = hftsvd.mylsvd()
lda_svd.initializeModel(metadata,lda.theta,flag=1)
lda_svd.train_sgd()
rmse = hftsvd.RMSE(lda_svd,metadata)
print "LDA->SVD:%f"%rmse.compute()
Corpus_r = parse.amaCorpus()
Corpus_r.set_data(metadata,form='review')
lda_review = hftlda.myllda(result_path=path+'review/')
lda_review.initializeModel(Corpus_r)
lda_review.inferenceModel()
newtheta = hftlda.hweighted_theta(lda_review,metadata)
hsvd = hftsvd.mylsvd()
hsvd.initializeModel(metadata,newtheta,flag=1)
hsvd.train_sgd()
rmse = hftsvd.RMSE(hsvd,metadata)
print "HWeighted SVD RMSE:%f"%rmse.compute()
wsvd = mylwsvd.WSVD()
wsvd.initializeModel(metadata,lda_review.theta)
wsvd.train_sgd()
rmse = hftsvd.RMSE(wsvd,metadata)
print "HLMF RMSE:%f"%rmse.compute()
'''