import os
import numpy as np
import pandas as pd
import json
import re
import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
from gensim import models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support 
import dill
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from bert_serving.client import BertClient

class classifier_mlp:
	# def __init__(self, word2vec, json_file):
	# 	# model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	# 	self.word2vec=models.KeyedVectors.load_word2vec_format(word2vec,binary=True)
	# 	self.data=pd.read_json(json_file, lines=True)
	def __init__(self, json_file):
		# model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
		# self.word2vec=models.KeyedVectors.load_word2vec_format(word2vec,binary=True)
		self.data=pd.read_json(json_file, lines=True)
		self.bc=BertClient(ip='localhost')
		self.csvFile=open("csvTrainVecBert.csv","w",newline='')
		self.writer=csv.writer(self.csvFile)
		
	def get_w2v_word(self, word):
		return (self.word2vec[word])
	def get_data_column(self, columns):
		return (self.data[columns])
	def preprocessing(self):
		self.data['text']=["title "+ str(i)+" " for i in range(0,len(self.data['short_description']))]+self.data.short_description
		# self.data['text']=self.data.headline+ " "+self.data.short_description
	def clean_text(self):
		stemmer = SnowballStemmer('english')
		words = stopwords.words('english')
		self.data['cleaned_text'] = self.data['text'].apply(lambda x:" ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]"," ",x).split() if i not in words]))
		return self.data['cleaned_text']
	# def compute_doc_vec_single(self, clean_text):
	# 	vec = np.zeros((self.word2vec.vector_size,), dtype=np.float32)
	# 	n = 0
	# 	tokenized_clean_text=nltk.word_tokenize(clean_text)
	# 	for word in tokenized_clean_text:
	# 		if word in self.word2vec:
	# 			vec += self.word2vec[word]
	# 			n += 1
	# 	if(n==0):
	# 		return (self.word2vec["Hello"]*0)
	# 	else:
	# 		return (vec/n)
	def compute_doc_vec_single_bert(self, clean_text):
		bert_sen_vec=self.bc.encode([clean_text])
		self.writer.writerow(bert_sen_vec[0])
		# csvFile=open("csvTrainVecBert.csv","w",newline='')
		# writer=csv.writer(csvFile)
		# writer.writerow(bert_sen_vec[0])
		# csvFile.close()
		return (bert_sen_vec[0])

	def compute_doc_vec(self, clean_text):
		print("Getting the bert vector...")
		return np.row_stack([self.compute_doc_vec_single_bert(x) for x in clean_text])

	def get_target_y(self):
		y_encoder=LabelEncoder()
		y=y_encoder.fit_transform(self.data['category'])
		return y,y_encoder
	def get_train_test(self, x, y):
		train_idx, test_idx = train_test_split(range(len(y)), test_size=0.1, stratify=y, random_state=1)
		train_x=x[train_idx, :]
		train_y=y[train_idx]
		test_x=x[test_idx, :]
		test_y=y[test_idx]
		return train_x, train_y, test_x, test_y,train_idx,test_idx
	def get_MLP_model(self, train_x, train_y):
		X=train_x
		Y=train_y
		model = MLPClassifier(hidden_layer_sizes=(256, 256), activation='relu', solver='adam',batch_size=4,max_iter=300, shuffle=True,verbose=1)
		model.fit(X, Y)
		print ("Neural network layers(hidden layers included): ",model.n_layers_)
		print ("Iterations: ",model.n_iter_)
		print ("Loss: ",model.loss_)
		print ("Output layer activation function: ",model.out_activation_)
		return (model)
	def get_csv_file(self, train_idx,test_idx, train_y,test_y):
		csvFile=open("csvTest1.csv", "w", newline='')
		writer=csv.writer(csvFile)
		#writer.writerow(["short_description","category"])
		for i in range(0,len(test_idx)):
			writer.writerow([test_y[i],self.data['cleaned_text'][test_idx[i]]])
		csvFile.close()

	def save_model(self, model, y_encoder):
		output_dir=u'MLP_bert_output'
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		model_file=os.path.join(output_dir, u'model.pkl')
		with open(model_file, 'wb') as outfile:
			pickle.dump({'y_encoder': y_encoder, 'mlp': model}, outfile)
		return model_file

	def CV_predict(self,x,y):
		print("10-fold cross validation...")
		score=cross_val_score(MLPClassifier(hidden_layer_sizes=(32, 32), activation='relu', solver='adam',batch_size=16,max_iter=80, shuffle=True,verbose=1), x,y, cv=10)
		print(np.mean(score))
	

	def tradition_predict(self, mlp_model_file,test_idx, truth):
		print("Traditional predict method...")
		clf_model=Predictor(mlp_model_file)
		new_y_pred = clf_model.predict(self.data['cleaned_text'][test_idx])
		df=pd.DataFrame({u'test': new_y_pred, u'truth': truth[test_idx]})
		test=list(map(str,df['test']))
		truth=list(map(str,df['truth']))
		count=0
		for i in range(0, len(test)):
			if test[i]==truth[i]:
				count+=1
		print(count/len(test))
		clf_model._close_csvFile()
	def close_csvFile(self):
		self.csvFile.close()
	
	def process(self):
		print("Preprocessing...")
		self.preprocessing()
		print("Cleaning text...")
		temp_clean_text=self.clean_text()
		print("Getting the article vector...")
		x=self.compute_doc_vec(temp_clean_text)
		print("Getting train data and test data...")
		y, y_encoder=self.get_target_y()
		train_x, train_y, test_x, test_y, train_idx,test_idx=self.get_train_test(x,y)
		# print("*******************************")
		# self.get_csv_file(train_idx, test_idx,train_y,test_y)
		# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
		print("Getting model...")
		model=self.get_MLP_model(train_x, train_y)
		print("Saving model...")
		model_file=self.save_model(model, y_encoder)

		self.close_csvFile()
		return model_file, test_idx, x,y

# class Predictor(object):
# 	def __init__(self, loaded_w2v, mlp_model_file):
# 		self.w2v = loaded_w2v
# 		with open(mlp_model_file, 'rb') as infile:
# 			self.model = pickle.load(infile)
# 	def get_model(self):
# 		return (self.model)
# 	def predict(self, articles):
# 		x = self._compute_doc_vec(articles)
# 		y = self.model['mlp'].predict(x)
# #         y_label = self.model1['y_encoder'].inverse_transform(y)
# 		return y
    
# 	def _compute_doc_vec(self, clean_text):
# 		return np.row_stack([self._compute_doc_vec_single(x) for x in clean_text])

# 	def _compute_doc_vec_single(self, clean_text):
# 		vec = np.zeros((self.w2v.vector_size,), dtype=np.float32)
# 		n = 0
# 		tokenized_clean_text=nltk.word_tokenize(clean_text)
# 		for word in tokenized_clean_text:
# 			if word in self.w2v:
# 				vec += self.w2v[word]
# 				n += 1
# 		if(n==0):
# 			return (self.w2v["Hello"]*0)
# 		else:
# 			return (vec/n)
class Predictor(object):
	def __init__(self, mlp_model_file):
		self.bc=BertClient(ip='localhost')
		with open(mlp_model_file, 'rb') as infile:
			self.model = pickle.load(infile)
		self.csvFile1=open("csvTestBert.csv","w",newline='')
		self.writer1=csv.writer(self.csvFile1)
	def get_model(self):
		return (self.model)
	def predict(self, articles):
		x = self._compute_doc_vec(articles)
		y = self.model['mlp'].predict(x)
#         y_label = self.model1['y_encoder'].inverse_transform(y)
		return y
    
	def _compute_doc_vec(self, clean_text):
		print("Getting the bert vector...")
		return np.row_stack([self._compute_doc_vec_single_bert(x) for x in clean_text])

	def _compute_doc_vec_single_bert(self, clean_text):
		bert_sen_vec=self.bc.encode([clean_text])
		self.writer1.writerow(bert_sen_vec[0])
		# csvFile=open("csvTestVecBert.csv","w",newline='')
		# writer=csv.writer(csvFile)
		# writer.writerow(bert_sen_vec[0])
		# csvFile.close()
		return (bert_sen_vec[0])
	def _close_csvFile(self):
		self.csvFile1.close()

if __name__=="__main__":
	print("Loading the pre-trained the json file...")
	MLP_classifier=classifier_mlp("News_Category_Dataset_v2.json")
	clf, test_idx, x,y=MLP_classifier.process()


	#Two methods to get precision
	#Method1 tradition, the same as online test
	#**********************************************************
	MLP_classifier.tradition_predict(clf, test_idx,y)
	#**********************************************************
	
	#Method2 10-fold cross validation
	#**********************************************************
	# MLP_classifier.CV_predict(x,y)
	#**********************************************************


