#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import codecs
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

class word2vec(object):
    def __init__(self,raw_file_name='../../temp_results/raw_news_text.txt'):
        self.raw_file_name = raw_file_name
    
    def train_word_2_vec(self,model_save_file_name='../../temp_results/word2vec_hindi.txt'):
        model = Word2Vec(LineSentence(self.raw_file_name), size=300,workers=multiprocessing.cpu_count())
        model.wv.save_word2vec_format(model_save_file_name, binary=False)

    def pretty_print(self,main_word,list_of_words):
        file = codecs.open("../../temp_results/sample_result.txt", "a+", "utf-8")
        file.write(main_word)
        file.write("\n")
        for each_word in list_of_words:
            file.write(each_word[0])
            file.write(" ")
            file.write(str(each_word[1]))
            file.write("\n")
        file.write("\n=======================\n")
        file.close()

    def check_for_similar_words(self,):
        from gensim.models.keyedvectors import KeyedVectors
        model = KeyedVectors.load_word2vec_format("../../temp_results/word2vec_hindi.txt", binary=False)
        
        self.pretty_print(u"भारत",model.most_similar(u"भारत"))
        self.pretty_print(u"सिंह",model.most_similar(u"सिंह"))
        self.pretty_print(u"क्रिकेट",model.most_similar(u"क्रिकेट"))
        self.pretty_print(u"रुपये",model.most_similar(u"रुपये"))
        
        
def main():
    w2v = word2vec()
    w2v.train_word_2_vec()
    
if __name__ == "__main__":
    main()