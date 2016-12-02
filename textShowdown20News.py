from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from time import time
import numpy as np
from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import scipy.sparse as ss
import copy

# older version for numpy matrix input
# output numpy matrix
# slow
def dataToTfidf(vectorResult):
	n = len(vectorResult)
	m = len(vectorResult[0])
	tfmatrix = np.zeros((n,m))
	idfmatrix = np.zeros(m)
	# term t to document d
	for d in xrange(n):
		for t in xrange(m):
			if(vectorResult[d][t] <= 0):
				continue
			tfmatrix[d][t] = 1. + np.log(vectorResult[d][t])

	for t in xrange(m):
		for d in xrange(n):
			if(vectorResult[d][t] <= 0):
				continue
			idfmatrix[t] += 1
		# special case for the most frequent word
		if(idfmatrix[t] >= n-1):
			idfmatrix[t] -= 1
	# print "check negative..."
	# for i in range(m):
	# 	if(idfmatrix[i]<0):
	# 		print i,' ',"failed ",idfmatrix[i]
	# print "check negative completed"
	idfmatrix = np.log(n/(idfmatrix + 1.))

	tfidf = list(tfmatrix * idfmatrix)
	tfidfnorm = list(np.linalg.norm(tfidf,axis=1))
	X = [tfidf[i]/tfidfnorm[i] for i in xrange(n)]
	return X

# input is a scipy sparse matrix
# should return a sparse matrix
# fast
def sparseToTfidf(spaMat):
	spaMat.data = np.array([float(i) for i in spaMat.data])

	tfMatrix = copy.deepcopy(spaMat)
	tfMatrix = tfMatrix.log1p()
	tfMatrix.data += 1.

	n,d = spaMat.shape
	idftmp = np.array([0]*d)
	for index,col in enumerate(spaMat.indices):
		idftmp[col] += spaMat.data[index]
	idftmp = np.log(sum(idftmp)/(idftmp + 1.))
	idfMatrix = ss.csr_matrix(idftmp)
	tfidfMatrix = tfMatrix.multiply(idfMatrix)

	# normalize
	for row in range(len(tfidfMatrix.indptr)-1):
		l2norm = sum(tfidfMatrix.data[tfidfMatrix.indptr[row]:tfidfMatrix.indptr[row+1]]**2)**0.5
		tfidfMatrix.data[tfidfMatrix.indptr[row]:tfidfMatrix.indptr[row+1]] /= l2norm
	return tfidfMatrix

# categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset = 'train',shuffle=True,random_state=42)
twenty_test = fetch_20newsgroups(subset = 'test',shuffle=True,random_state=42)

# print len(twenty_train.data),len(twenty_test.data)
# D = len(bayesCounter)
# wordClassNum = len(bayesCounter)
# datatable = [[None for _ in range(len(twenty_train.target_names))] for _ in range(wordClassNum)]
tfidf_vectorizer = CountVectorizer(lowercase=True,strip_accents='ascii',stop_words='english')
vectorResult = tfidf_vectorizer.fit_transform(twenty_train.data[:10000])
vectorResult2 = tfidf_vectorizer.transform(twenty_test.data[:4000])

# tfidf_vectorizer2 = TfidfVectorizer(lowercase=True,strip_accents='ascii',stop_words='english',sublinear_tf=True,norm='l2')
# tfidf2 = tfidf_vectorizer2.fit_transform(twenty_train.data[0:10000]).toarray()
# test_vectorizer2 = TfidfVectorizer(lowercase=True,strip_accents='ascii',stop_words='english',sublinear_tf=True,norm='l2')
# testidf2 = test_vectorizer2.transform(twenty_test.data[0:3000]).toarray()
# print sum(sum(tfidf2))

X = sparseToTfidf(vectorResult[:10000])
Xtest = sparseToTfidf(vectorResult2[:4000])
y = twenty_train.target[:10000]
ytest = twenty_test.target[:4000]

clf_NB = MultinomialNB()
t0 = time()
clf_NB.fit(X,y)
t_NB = time()-t0
ypredict_NB = clf_NB.predict(Xtest)
ypredict_NB_train = clf_NB.predict(X)

clf_SVM_Cosine = SVC(kernel=cosine_similarity)
t0 = time()
clf_SVM_Cosine.fit(X,y)
t_SVM = time()-t0
ypredcit_SVM = clf_SVM_Cosine.predict(Xtest)
ypredcit_SVM_train = clf_SVM_Cosine.predict(X)

accuracy_NB = accuracy_score(ypredict_NB,ytest)
accuracy_NB_train = accuracy_score(ypredict_NB_train,y)
accuracy_SVM = accuracy_score(ypredcit_SVM,ytest)
accuracy_SVM_train = accuracy_score(ypredcit_SVM_train,y)

precision_NB,recall_NB,_,_ = precision_recall_fscore_support(ytest,ypredict_NB)
precision_NB_train,recall_NB_train,_,_ = precision_recall_fscore_support(y,ypredict_NB_train)
precision_SVM,recall_SVM,_,_ = precision_recall_fscore_support(ytest,ypredcit_SVM)
precision_SVM_train,recall_SVM_train,_,_ = precision_recall_fscore_support(y,ypredcit_SVM_train)

precision_NB = sum(precision_NB)/len(precision_NB)
precision_NB_train = sum(precision_NB_train)/len(precision_NB_train)
recall_NB = sum(recall_NB)/len(recall_NB)
recall_NB_train = sum(recall_NB_train)/len(recall_NB_train)

precision_SVM = sum(precision_SVM)/len(precision_SVM)
precision_SVM_train = sum(precision_SVM_train)/len(precision_SVM_train)
recall_SVM = sum(recall_SVM)/len(recall_SVM)
recall_SVM_train = sum(recall_SVM_train)/len(recall_SVM_train)

print "Naive Bayes"
print "\t\tTest\t\tTrain"
print "Accuracy: \t%.8f"%accuracy_NB,'\t',str(accuracy_NB_train)
print "Precision: \t"+str(precision_NB),'\t',str(precision_NB_train)
print "Recall: \t"+str(recall_NB),'\t',str(recall_NB_train)
print "Training time: %.2fs"%t_NB

print "\nSVM with cosine kernel"
print "\t\tTest\t\tTrain"
print "Accuracy: \t%.8f"%accuracy_SVM,'\t',str(accuracy_SVM_train)
print "Precision: \t"+str(precision_SVM),'\t',str(precision_SVM_train)
print "Recall: \t"+str(recall_SVM),'\t',str(recall_SVM_train)
print "Training time: %.2fs"%t_SVM



