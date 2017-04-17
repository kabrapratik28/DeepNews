# -*- coding: utf-8 -*-
import codecs
import os
import numpy as np
import pandas as pd
import cPickle as pickle
from tqdm import tqdm
from collections import Counter
from gensim import corpora
from six import iteritems
from gensim import corpora,models,similarities
#warning off
pd.options.mode.chained_assignment = None

class tf_idf(object):

    def __init__(self):
        self.dictionary = None
        self.tfidf_model = None

    def file_line_counter(self,file_name):
        """
        Copy of code from model.py file !!! Try to DRY :) 
        """
        with codecs.open(file_name, 'r',encoding='utf8') as f:
            for i, l in tqdm(enumerate(f)):
                pass
        print (file_name," contains ",i+1," lines.")
        return i+1

    def train_tfidf_model(self,file_path='../../temp_results/corpus.txt'):
        textfile = codecs.open(file_path, "r", "utf-8")   
        
        print("Reading and Processing Text File")
        first_lines=[]
        for line in textfile:
            first_lines.append(line.strip())
        
        print ("--------Building Corpora Dictionary---------------" )
        dictionary = corpora.Dictionary(line.split('#|#')[1].split() for line in first_lines)
        
        #remove words that appear less than 2 times
        #twoids = [tokenid for tokenid,docfreq in iteritems(dictionary.dfs) if docfreq < 2]
        #dictionary.filter_tokens(fiveids)
        
        #Remove Gaps
        dictionary.compactify()
        dictionary.save_as_text('../../temp_results/tfidf_dictionary.txt',sort_by_word=False)
        dictionary.save('../../temp_results/tfidf_dictionary')
        print("Dictionary Saved")
                    
        print ("--Now Transforming to Bag of Words Vectors on the Fly--")
        class MyCorpus(object):
            def __iter__(self):
                for line in first_lines:
                    yield dictionary.doc2bow(line.split()) 
                
        news_corpus  = MyCorpus()
        print("Corpus Built...Now Starting Model Training")
        tfidf_model = models.TfidfModel(news_corpus)
        tfidf_model.save('../../temp_results/tfidf_model')
        print("Model Trained & Saved")

    def load_model_and_dictionary(self):
        self.tfidf_model = models.TfidfModel.load('../../temp_results/tfidf_model')
        self.dictionary = corpora.Dictionary.load('../../temp_results/tfidf_dictionary')
        print ("Dictionary & Model Loaded Successfully")     

    def generate_tf_idf_scores(self,test_document_words):
        #Convert test document to bag of words
        test_bow =  self.dictionary.doc2bow(test_document_words)
        tfidf_corpus = self.tfidf_model[test_bow]
        tfidf_scores = [(self.dictionary.get(tokenid),score) for (tokenid,score) in tfidf_corpus]        
        return tfidf_scores
    
    def tf_idf_imp_words(self,sentence,word2_vec_set,top_k):
        document = sentence.strip().split()
        if len(document)<=top_k:
            return u" ".join(document)
        
        tf_idf_scores = self.generate_tf_idf_scores(document)
        np_tf_idf_scores = np.array(tf_idf_scores)
        data = pd.DataFrame({"word":np_tf_idf_scores[:,0],"tf_idf":np_tf_idf_scores[:,1]})
        word_freq = Counter(document)

        data['freq'] = data['word'].map(word_freq)
        data.sort_values(['tf_idf'],ascending=False,inplace=True)
        data_filtered = data[ data['word'].isin( word2_vec_set) ]
        data_filtered['cum_freq']= data_filtered['freq'].cumsum()
        cum_sum_series = data_filtered[data_filtered['cum_freq'] >= top_k]['cum_freq']

        if len(cum_sum_series) > 0:
            cut_off = cum_sum_series.iloc[0]
            data_filtered = data_filtered[data_filtered['cum_freq'] <= cut_off]
            if  cut_off != top_k and len(data_filtered['cum_freq'])>2:
                data_filtered.iloc[len(data_filtered['cum_freq'])-1,data_filtered.columns.get_loc('freq')]=top_k-data_filtered.iloc[len(data_filtered['cum_freq'])-2]['cum_freq']
        
        word_freq_array = data_filtered[['word','freq']].values
        dic_final ={}
        for (word,freq) in word_freq_array:
            dic_final[word]=freq
        list_to_return = []
        for word in document:
            if word in dic_final and dic_final[word]>0:
                list_to_return.append(word)
                dic_final[word]-=1
        return u" ".join(list_to_return)
    
    def process_file(self,file_to_process, processed_file_name,word2_vec_set,top_k,seperator="#|#"):
        file_lines = self.file_line_counter(file_to_process)
        file_to_process_fp = codecs.open(file_to_process,encoding='utf-8')
        with codecs.open(processed_file_name, "w", "utf-8") as fp:            
            for i in tqdm(range(file_lines)):
                news_data = file_to_process_fp.readline().strip()
                headline, desc = news_data.split(seperator)
                processed_line = self.tf_idf_imp_words(desc,word2_vec_set,top_k)
                fp.write(headline+seperator+processed_line+"\n")
        file_to_process_fp.close()
        
if __name__ == '__main__':
    
    word2vec_file_name = '../../temp_results/word2vec_hindi.txt'
    file_to_process = '../../temp_results/train_corpus.txt'
    processed_file_name = '../../temp_results/tfidf_based_train_corpus.txt'
    #generate top 100000 words from word2vec ... 
    top_lines = 100000
    #top words to consider in desc
    top_k = 50
    #read word2vec file
    word2vec_fp = codecs.open(word2vec_file_name,encoding='utf-8')
    dimension_line = word2vec_fp.readline()
    
    word2vec_top_k_set = set()
    for i in range(top_lines):
        each_word_dimension = word2vec_fp.readline().strip()
        data = each_word_dimension.split()
        word = data[0]
        word2vec_top_k_set.add(word)
    
    word2vec_fp.close()
    print ("word2vec top {} words loaded in set".format(top_lines))
    
    ti = tf_idf()
    ti.load_model_and_dictionary()
    ti.process_file(file_to_process,processed_file_name,word2vec_top_k_set,top_k)
    