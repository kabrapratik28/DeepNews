import numpy as np
import random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
seed = 21
random.seed(seed)
np.random.seed(seed)

#number of validation examples
nb_val_samples=3000
embedding_dimension = 300
max_length=25
rnn_layers = 3
rnn_size = 512

class train(object):
    def __init__(self,):
        self.word2vec = None
        self.idx2word = {}
        self.word2idx = {}
        
        #initalize end of sentence and empty
        self.word2idx['<empty>'] = 0
        self.word2idx['<eos>'] = 1
        self.word2idx[0] = '<empty>'
        self.word2idx[1] = '<eos>'
        
        #TODO: ADD <empty>, <eos> vectors
        #TODO: store/load this dictionaries from pickle
        
    def read_word_embedding(self,file_name='../../temp_results/word2vec_hindi_sample.txt'):
        """
        read word embedding file and assign indexes to word
        """
        idx = 2
        temp_word2vec_dict = {}
        with open(file_name) as fp:
            for each_line in fp:
                word_embedding_data = each_line.split(" ")
                word = word_embedding_data[0]
                vector = [float(i) for i in word_embedding_data[1:]]
                temp_word2vec_dict[idx] = vector
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx = idx + 1
                if idx%10000 == 0:
                    print ("working on word2vec ... idx ",idx)
                    
        #TODO: <empty>, <eos> vectors initializations?
        
        length_vocab = len(temp_word2vec_dict)
        shape = (length_vocab,embedding_dimension)
        #faster initlization and random for <empty> and <eos> tag
        self.word2vec = np.random.uniform(low=-1, high=1, size=shape)
        for i in range(length_vocab):
            if i in temp_word2vec_dict:
                self.word2vec[i,:] = temp_word2vec_dict[i]
        
    def sentence2idx(self,sentence, max_length=25):
        """
        given a sentence convert it to its ids
        "I like India" => [12, 51, 102]
        words not present in vocab igonre them
        """
        list_idx = []
        tokens = sentence.split(" ")
        count = 0 
        for each_token in tokens:
            if each_token in self.word2idx:
                list_idx.append(self.word2idx[each_token])
                count = count + 1
                if count>=max_length:
                    break
        
        #TODO: left padding and right padding according to
        #head line or desc
        
        return list_idx
    
    def read_data_files(self,file_name='../../temp_results/raw_news_text.txt',seperator='#|#'):
        """
        Assumes one line contatin "headline seperator description"
        """
        X = []
        y = []
        with open(file_name) as fp:
            for each_line in fp:
                each_line = each_line.strip()
                headline, desc = each_line.split(seperator)
                headline_idx = self.sentence2idx(headline)
                desc_idx = self.sentence2idx(desc)
                X.append(headline_idx)
                y.append(desc_idx)
        return (X,y)
    
    def split_test_train(self,X,y):
        """
        split X,y data into training and testing
        """
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=nb_val_samples, random_state=seed)
        return (X_train, X_test, Y_train, Y_test)

    
    def create_model(self,):
        
        length_vocab, embedding_size = self.word2vec.shape
        print ("shape of word2vec matrix ",self.word2vec.shape)
        
        model = Sequential()
        
        #TODO: look at mask zero flag
        model.add(
                Embedding(
                        length_vocab, embedding_size,
                        input_length=max_length,
                        weights=[self.word2vec], mask_zero=True,
                        name='embedding_layer'
                )
        )
        
        for i in range(rnn_layers):
            lstm = LSTM(rnn_size, return_sequences=True,
                dropout=0, recurrent_dropout=0,
                name='lstm_layer_%d'%(i+1)
            )
            
            model.add(lstm)
            #No drop out added !
            
        return model

if __name__ == '__main__':
    t = train()
    t.read_word_embedding()
    X,y = t.read_data_files()
    #TODO: Testing purpose
    nb_val_samples=3
    X_train, X_test, Y_train, Y_test  =t.split_test_train()