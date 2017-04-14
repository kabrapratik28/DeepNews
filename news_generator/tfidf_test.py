# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:46:47 2017

@author: Vatsal Shah
"""

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



raw_file_name='combined_annotated_news_text.txt'
temp_results='../../temp_results'
file_path = os.path.join(temp_results,raw_file_name)
textfile = codecs.open(file_path, "r", "utf-8")   
count = 0

first_lines=[]
for line in textfile:
    if count<100:
        first_lines.append(line)
    else:
        break
    count+=1

train = first_lines[0:50]
test = first_lines[51:]

print ("--------Building Corpora Dictionary---------------" )
dictionary = corpora.Dictionary(line.split('#|#')[1].split() for line in train)

#remove words that appear less than 2 times
twoids = [tokenid for tokenid,docfreq in iteritems(dictionary.dfs) if docfreq < 2]
#dictionary.filter_tokens(fiveids)

#Remove Gaps
dictionary.compactify()
dictionary.save_as_text('../../temp_results/tfidf_dictionary.txt',sort_by_word=False)
dictionary.save('../../temp_results/tfidf_dictionary')
#print (dictionary.token2id)

print (" --------------- Dictionary Built and Saved.. Now Transforming to Bag of Words Vectors on the Fly----------")
class MyCorpus(object):
    def __iter__(self):
        for line in train:
            yield dictionary.doc2bow(line.split()) 


news_corpus  = MyCorpus()
#print(news_corpus)

#for vector in news_corpus:
#    print (vector)
print("------------------Corpus Built...Now Starting Model Training-------------")

tfidf = models.TfidfModel(news_corpus)
tfidf.save('../../temp_results/tfidf_model')
print("---------------------------Model Trained-----------------------")


if (os.path.exists("../../temp_results/tfidf_model")):
    tfidf = models.TfidfModel.load('../../temp_results/tfidf_model')
    dictionary = corpora.Dictionary.load('../../temp_results/tfidf_dictionary')
    print ("Dictionary & Model Loaded Successfully")
else :
    print ("Could not find Dictionary & Model!")
    


test_document =  dictionary.doc2bow(test[0].split(' '))

tfidf_corpus = tfidf[test_document]

temp = [(dictionary.get(tokenid),score) for (tokenid,score) in tfidf_corpus]        




