# -*- coding: utf-8 -*-
import codecs
import os
import numpy as np
import pandas as pd
import cPickle as pickle

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

    def train_tfidf_model(self,raw_file_name='combined_annotated_news_text.txt',temp_results='../../temp_results'):
        file_path = os.path.join(temp_results,raw_file_name)
        textfile = codecs.open(file_path, "r", "utf-8")   
        
        print("Reading and Processing Text File")
        first_lines=[]
        for line in textfile:
            first_lines.append(line)
            
        #    train_indices = int(0.8*(len(first_lines)))
        #    train = first_lines[0:train_indices]
        #    test = first_lines[train_indices:]
             
        
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
        
            #print (dictionary.token2id)
            
        print (" -- Dictionary Built and Saved.. Now Transforming to Bag of Words Vectors on the Fly--")
        class MyCorpus(object):
            def __iter__(self):
                for line in first_lines:
                    yield dictionary.doc2bow(line.split()) 
        
        
        news_corpus  = MyCorpus()
        #print(news_corpus)
        
        #for vector in news_corpus:
        #    print (vector)
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
        
if __name__ == '__main__':

    raw_file_name='combined_annotated_news_text.txt'
    temp_results='../../temp_results'
    file_path = os.path.join(temp_results,raw_file_name)
    textfile = codecs.open(file_path, "r", "utf-8") 
    count=0
    test_document=[]
    for line in textfile:
        if count<5:
            test_document.append(line)
        else:
            break
        count+=1
    
    word2_vec_set = pickle.load( open( "../../temp_results/b.pickle", "rb" ) ) 
    
    ti = tf_idf()
    ti.load_model_and_dictionary()
    for sentence in test_document:
        print ti.tf_idf_imp_words(sentence.split("#|#")[1],word2_vec_set,50)