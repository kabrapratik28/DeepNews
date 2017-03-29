# -*- coding: utf-8 -*-
import unittest
import codecs
import numpy as np
import pandas as pd

class SimplisticTest(unittest.TestCase):
    def test_model(self,):
        tokens=set()
        with codecs.open('../../temp_results/raw_news_text.txt','r',encoding='utf8') as fp:
            for each_line in fp:
                headline, text = each_line.split("#|#")
                tokens.update(headline.split())
                tokens.update(text.split())
        tokens_word2vec = np.random.rand(len(tokens),300)
        df = pd.DataFrame({'word':list(tokens)})
        shape = tokens_word2vec.shape
        for i in range(shape[1]):
            df[i] = tokens_word2vec[:,i]
        print ("sample word2vec writting ...")
        df.to_csv('../../temp_results/word2vec_hindi.txt',sep=' ',header=None, index=False, encoding='utf-8')

if __name__ == '__main__':
    unittest.main()