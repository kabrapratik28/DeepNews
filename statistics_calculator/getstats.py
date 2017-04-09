#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import codecs
import os
import numpy as np
import matplotlib.pyplot as plt

def getstats(temp_results='../../temp_results',raw_file_name='raw_news_text.txt',n_words=20):
    file_path = os.path.join(temp_results,raw_file_name)
    textfile = codecs.open(file_path, "r", "utf-8")    
    
    word_freq_counter = {}
    for line in textfile:
        words = line.split(' ')
        for word in words:
            if word in word_freq_counter:
                word_freq_counter[word]+=1
            else:
                word_freq_counter[word]=1
    
    word_freq_sorted = [(w, word_freq_counter[w]) for w in sorted(word_freq_counter, key=word_freq_counter.get, reverse=True)]
    
    most_frequent_words = word_freq_sorted[0:n_words]
    
    words = [u" "+word for (word,freq) in most_frequent_words ]
    freq =  [freq for (word,freq) in most_frequent_words]        
        
    indexes = np.arange(len(words))
    plt.bar(indexes, freq)
    
    # add labels
    plt.xticks(indexes, words)
    plt.show()