#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 01:19:44 2017

@author: student
"""

import codecs
import os
import numpy as np
import matplotlib.pyplot as plt
#def getstats(temp_results='../../temp_results',raw_file_name='annotated_news_text.txt',n_words=20):

    
temp_results='../../'
raw_file_name='news_data.txt'
file_path = os.path.join(temp_results,raw_file_name)
textfile = codecs.open(file_path, "r", "utf-8")   

article_word_freq = {}
#headline_word_freq ={}


article_len = 0
count_articles=0

#lines = [line for line in textfile]
#first_lines = lines[0:5]

for line in textfile:
        
        tokens = line.split(' ')
        for words in tokens:
            if words in article_word_freq:
                article_word_freq[words]+=1
            else:
                article_word_freq[words]=1
        
        article_len += len(tokens)
        count_articles+=1
       
    

article_len/=count_articles


print "Total Unique Tokens in Articles: "+ str(len(article_word_freq))
print "Average Length of Article "+ str(article_len)

