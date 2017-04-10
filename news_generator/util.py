# -*- coding: utf-8 -*-
import codecs
import pandas as pd
from collections import Counter
from gensim import models

class preprocess(object):
    def top_k_freq_words(self,file_names,top_k,seperator="#|#",return_word_only=True):
        """
        Top k frequent occuring word in given file
        """
        c = Counter()
        for file_name in file_names:
            print ("Reading file ",file_name)
            with codecs.open(file_name, 'r',encoding='utf8') as fp:
                for each_line in fp:
                    each_line = each_line.strip()
                    each_line = each_line.replace(seperator, " ")
                    each_line = each_line.split()
                    c.update(each_line)
        most_common_words = c.most_common(top_k)
        if return_word_only:
            list_of_words = [x[0] for x in most_common_words]
            return list_of_words
        else:    
            return most_common_words

    def top_k_word2vec(self,word2vec_file_name,top_k_words,word2vec_dimension,new_file_name):
        """
        Given top k words return word2vec for that file
        top_k_words = [word1, word2, ...]
        """
        #word2vec = pd.read_csv("../../temp_results/a.txt",sep=' ', header=None, skiprows=range(1))
        model = models.KeyedVectors.load_word2vec_format(word2vec_file_name, binary=False)
        filtered_vectors = model[top_k_words]
        word2vec_frame = pd.DataFrame({'name':top_k_words})
        for i in range(word2vec_dimension):
            word2vec_frame[i] = filtered_vectors[:,i]
        word2vec_frame.to_csv(new_file_name,sep=" ",encoding='utf-8',index=False)

    def new_file_name(self,old_file_name,top_k):
        """
        create new file name for word2vec
        """
        return old_file_name.rstrip(".txt") + "_top_" + str(top_k) + ".txt"