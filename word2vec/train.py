#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class word2vec(object):
    def __init__(self,raw_file_name='../../temp_results/raw_news_text.txt'):
        self.raw_file_name = raw_file_name
    
    def train_word_2_vec(self,model_save_file_name='../../temp_results/word2vec_hindi.txt'):
        model = Word2Vec(LineSentence(self.raw_file_name), size=300,workers=multiprocessing.cpu_count())
        model.wv.save_word2vec_format(model_save_file_name, binary=False)

def main():
    w2v = word2vec()
    w2v.train_word_2_vec()
    
if __name__ == "__main__":
    main()