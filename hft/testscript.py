# testscript

import parse
import hftlda
import hftsvd
import profile

def profileTest():
	metadata = parse.amazonData()
	metadata.readdata('data/Arts.txt.gz')
	Corpus = parse.amaCorpus()
	Corpus.set_data(metadata,form='product')

	lda = hftlda.myllda()
	lda.initializeModel(Corpus)
	lda.inferenceModel()

	only_svd = hftsvd.mylsvd()
	ldatheta = hftsvd.randmat(metadata.M,only_svd.dim)
	only_svd.initializeModel(metadata,ldatheta,flag=0)
	only_svd.train_sgd()
	rmse = hftsvd.RMSE(only_svd,metadata)
	print "SVD ONLY RMSE:%f"%rmse.compute()

	lda_svd = hftsvd.mylsvd()
	lda_svd.initializeModel(metadata,lda.theta,flag=1)
	lda_svd.train_sgd()
	rmse = hftsvd.RMSE(lda_svd,metadata)
	print "LDA->SVD:%f"%rmse.compute()

if __name__=='__main__':
	profile.run("profileTest()")

