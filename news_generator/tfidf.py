# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:08:09 2017

@author: Vatsal Shah
"""
from gensim import corpora
import codecs
import os
import numpy as np
from six import iteritems
from gensim import corpora,models,similarities


def train_tfidf_model(raw_file_name='combined_annotated_news_text.txt',temp_results='../../temp_results'):
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
    twoids = [tokenid for tokenid,docfreq in iteritems(dictionary.dfs) if docfreq < 2]
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

    

def load_model_and_dictionary():
    tfidf_model = models.TfidfModel.load('../../temp_results/tfidf_model')
    dictionary = corpora.Dictionary.load('../../temp_results/tfidf_dictionary')
    print ("Dictionary & Model Loaded Successfully")     
    return tfidf_model,dictionary       

def generate_Tfidf_scores(tfidf_model,dictionary,test_document):
    #Convert test document to bag of words
    test_bow =  dictionary.doc2bow(test_document.split(' '))
    tfidf_corpus = tfidf_model[test_bow]
    tfidf_scores = [(dictionary.get(tokenid),score) for (tokenid,score) in tfidf_corpus]        
    return tfidf_scores
     
        
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
        
    
    # This function should be called only once
    train_tfidf_model(raw_file_name,temp_results)    
    
    # This function should be called only once
    tfidf_model,gen_dictionary=load_model_and_dictionary() 
    
    # Pass every article as a list (for eg: ["जानें मकर संक्रांति का पुण्यकाल मुहूर्त "])       
    scores=generate_Tfidf_scores(tfidf_model,gen_dictionary,test_document[0])

