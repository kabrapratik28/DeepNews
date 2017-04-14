# -*- coding: utf-8 -*-
import random
import codecs
import math
import time
import sys
import subprocess
import os.path
import cPickle as pickle
import numpy as np

import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Activation
from keras.utils import np_utils
from keras.preprocessing import sequence

from tqdm import tqdm
from sklearn.cross_validation import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from numpy import inf
from operator import itemgetter

from util import preprocess

"""
Tutorial on Keras Backend
=========================
import numpy as np
p=np.array([[[1,2],],])
q=np.array([ [[1,2],[3,4]] ])
kvar = K.variable(value=p, dtype='float64', name='example_var')
kvar2 = K.variable(value=q, dtype='float64', name='example_var')
activation_energies = K.batch_dot(kvar,kvar2, axes=(2,2))
print p.shape
print q.shape
print K.int_shape(activation_energies)
print activation_energies.eval()
"""


seed = 28
random.seed(seed)
np.random.seed(seed)

top_freq_word_to_use = 40000
embedding_dimension = 300
max_len_head = 25
max_len_desc = 50
max_length = max_len_head + max_len_desc
rnn_layers = 4
rnn_size = 600
# first 40 numebers from hidden layer output used for
# simple context calculation
activation_rnn_size = 50

empty_tag_location = 0
eos_tag_location = 1
unknown_tag_location = 2
learning_rate = 1e-4

class news_rnn(object):
    def __init__(self,):
        self.word2vec = None
        self.idx2word = {}
        self.word2idx = {}

        # initalize end of sentence, empty and unk tokens
        self.word2idx['<empty>'] = empty_tag_location
        self.word2idx['<eos>'] = eos_tag_location
        self.word2idx['<unk>'] = unknown_tag_location
        self.idx2word[empty_tag_location] = '<empty>'
        self.idx2word[eos_tag_location] = '<eos>'
        self.idx2word[unknown_tag_location] = '<unk>'
        # TODO: make model as part of self.model
        # TODO: store/load this dictionaries from pickle
    
    def file_line_counter(self,file_name):
        with codecs.open(file_name, 'r',encoding='utf8') as f:
            for i, l in tqdm(enumerate(f)):
                pass
        print (file_name," contains ",i+1," lines.")
        return i+1
    
    def read_word_embedding(self, file_name):
        """
        read word embedding file and assign indexes to word
        """
        idx = 3
        temp_word2vec_dict = {}
        # <empty>, <eos> tag replaced by word2vec learning
        # create random dimensional vector for empty, eos and unk tokens
        temp_word2vec_dict['<empty>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
        temp_word2vec_dict['<eos>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
        temp_word2vec_dict['<unk>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
        
        with codecs.open(file_name,'r',encoding='utf8') as fp:
            #skip first line of word embedding as it contains following information
            header = fp.readline().strip()
            header = header.split()
            if len(header)==2:
                print("number of words and dimesions ",header)
            else:
                print("number of dimesions ",len(header)-1)
            for each_line in fp:
                word_embedding_data = each_line.strip().split(" ")
                word = word_embedding_data[0]
                vector = [float(i) for i in word_embedding_data[1:]]
                temp_word2vec_dict[idx] = vector
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx = idx + 1
                if idx % 10000 == 0:
                    print ("working on word2vec ... idx ", idx)

        length_vocab = len(temp_word2vec_dict)
        shape = (length_vocab, embedding_dimension)
        # faster initlization and random for <empty> and <eos> tag
        self.word2vec = np.random.uniform(low=-1, high=1, size=shape)
        for i in range(length_vocab):
            if i in temp_word2vec_dict:
                self.word2vec[i, :] = temp_word2vec_dict[i]

    def padding(self, list_idx, curr_max_length, is_left):
        """
        padds with <empty> tag in left side
        """
        if len(list_idx) >= curr_max_length:
            return list_idx
        number_of_empty_fill = curr_max_length - len(list_idx)
        if is_left:
            return [empty_tag_location, ] * number_of_empty_fill + list_idx
        else:
            return list_idx + [empty_tag_location, ] * number_of_empty_fill

    def headline2idx(self, list_idx, curr_max_length, is_input):
        """
        if space add <eos> tag in input case, input size = curr_max_length-1
        always add <eos> tag in predication case, size = curr_max_length
        always right pad
        """
        if is_input:
            if len(list_idx) >= curr_max_length - 1:
                return list_idx[:curr_max_length - 1]
            else:
                # space remaning add eos and empty tags
                list_idx = list_idx + [eos_tag_location, ]
                return self.padding(list_idx, curr_max_length - 1, False)
        else:
            # always add <eos>
            if len(list_idx) == curr_max_length:
                list_idx[-1] = eos_tag_location
                return list_idx
            else:
                # space remaning add eos and empty tags
                list_idx = list_idx + [eos_tag_location, ]
                return self.padding(list_idx, curr_max_length, False)

    def desc2idx(self, list_idx, curr_max_length):
        """
        always left pad and eos tag to end
        """
        # desc padded left
        list_idx = self.padding(list_idx, curr_max_length, True)
        # eos tag add
        list_idx = list_idx + [eos_tag_location, ]
        return list_idx


    def sentence2idx(self, sentence, is_headline, curr_max_length, is_input=True):
        """
        given a sentence convert it to its ids
        "I like India" => [12, 51, 102]
        words not present in vocab igonre them

        is_input is only for headlines
        """
        list_idx = []
        tokens = sentence.split(" ")
        count = 0
        for each_token in tokens:
            if each_token in self.word2idx:
                list_idx.append(self.word2idx[each_token])
            else:
                #append unk token as original word not present in word2vec
                list_idx.append(self.word2idx['<unk>'])
            count = count + 1
            if count >= curr_max_length:
                break

        if is_headline:
            return self.headline2idx(list_idx, curr_max_length, is_input)
        else:
            return self.desc2idx(list_idx, curr_max_length)

    def split_test_train(self, X, y):
        """
        split X,y data into training and testing
        """
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=nb_val_samples, random_state=seed)
        return (X_train, X_test, Y_train, Y_test)

    def output_shape_simple_context_layer(self, input_shape):
        """
        Take input shape tuple and return tuple for output shape
        Output shape size for simple context layer =
        remaining part after activatoion calculation fron input layers avg. +
        remaining part after activatoion calculation fron current hidden layers avg.

        that is 2 * (rnn_size - activation_rnn_size))

        input_shape[0] = batch_size remains as it is
        max_len_head = heading max length allowed
        """
        return (input_shape[0], max_len_head , 2 * (rnn_size - activation_rnn_size))

    def simple_context(self, X, mask):
        """
        Simple context calculation layer logic
        X = (batch_size, time_steps, units)
        time_steps are nothing but number of words in our case.
        """
        # segregrate heading and desc
        desc, head = X[:, :max_len_desc, :], X[:, max_len_desc:, :]
        # segregrate activation and context part
        head_activations, head_words = head[:, :, :activation_rnn_size], head[:, :, activation_rnn_size:]
        desc_activations, desc_words = desc[:, :, :activation_rnn_size], desc[:, :, activation_rnn_size:]

        # p=(bacth_size, length_desc_words, rnn_units)
        # q=(bacth_size, length_headline_words, rnn_units)
        # K.dot(p,q) = (bacth_size, length_desc_words,length_headline_words)
        activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))

        # make sure we dont use description words that are masked out
        activation_energies = activation_energies + -1e20 * K.expand_dims(1. - K.cast(mask[:, :max_len_desc], 'float32'), 1)

        # for every head word compute weights for every desc word
        activation_energies = K.reshape(activation_energies, (-1, max_len_desc))
        activation_weights = K.softmax(activation_energies)
        activation_weights = K.reshape(activation_weights, (-1, max_len_head, max_len_desc))

        # for every head word compute weighted average of desc words
        desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
        return K.concatenate((desc_avg_word, head_words))

    def create_model(self,):
        """
        RNN model creation
        Layers include Embedding Layer, 3 LSTM stacked,
        Simple Context layer (manually defined),
        Time Distributed Layer
        """
        length_vocab, embedding_size = self.word2vec.shape
        print ("shape of word2vec matrix ", self.word2vec.shape)

        model = Sequential()

        # TODO: look at mask zero flag
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
                name='lstm_layer_%d' % (i + 1)
            )

            model.add(lstm)
            # No drop out added !

        model.add(Lambda(self.simple_context,
                     mask=lambda inputs, mask: mask[:, max_len_desc:],
                     output_shape=self.output_shape_simple_context_layer,
                     name='simple_context_layer'))

        vocab_size = self.word2vec.shape[0]
        model.add(TimeDistributed(Dense(vocab_size,
                                name='time_distributed_layer')))
        
        model.add(Activation('softmax', name='activation_layer'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        K.set_value(model.optimizer.lr, np.float32(learning_rate))
        print (model.summary())
        return model

    def flip_words_randomly(self, description_headline_data, number_words_to_replace, model):
        """
        Given actual data i.e. description + eos + headline + eos
        1. It predicts news headline (model try to predict, sort of training phase)
        2. From actual headline, replace some of the words,
        with most probable predication word at that location
        3.return description + eos + headline(randomly some replaced words) + eos
        (take care of eof and empty should not get replaced)
        """
        if number_words_to_replace <= 0 or model == None:
            return description_headline_data

        # check all descrption ends with <eos> tag else throw error
        assert np.all(description_headline_data[:, max_len_desc] == eos_tag_location)

        batch_size = len(description_headline_data)
        predicated_headline_word_idx = model.predict(description_headline_data, verbose=0, batch_size=batch_size)
        copy_data = description_headline_data.copy()
        for idx in range(batch_size):
            # description = 0 ... max_len_desc-1
            # <eos> = max_len_desc
            # headline = max_len_desc + 1 ...
            random_flip_pos = sorted(random.sample(xrange(max_len_desc + 1, max_length), number_words_to_replace))
            for replace_idx in random_flip_pos:
                # Don't replace <eos> and <empty> tag
                if (description_headline_data[idx, replace_idx] == empty_tag_location or
                description_headline_data[idx, replace_idx] == eos_tag_location):
                    continue

                # replace_idx offset moving as predication doesnot have desc
                new_id = replace_idx - (max_len_desc + 1)
                prob_words = predicated_headline_word_idx[idx, new_id]
                word_idx = prob_words.argmax()
                # dont replace by empty location or eos tag location
                if word_idx == empty_tag_location or word_idx == eos_tag_location:
                    continue
                copy_data[idx, replace_idx] = word_idx
        return copy_data

    def convert_inputs(self, descriptions, headlines, number_words_to_replace, model,is_training):
        """
        convert input to suitable format
        1.Left pad descriptions with <empty> tag
        2.Add <eos> tag
        3.Right padding with <empty> tag after (desc+headline)
        4.input headline doesnot contain <eos> tag
        5.expected/predicated headline contain <eos> tag
        6.One hot endoing for expected output
        """
        # length of headlines and descriptions should be equal
        assert len(descriptions) == len(headlines)

        X, y = [], []
        for each_desc, each_headline in zip(descriptions, headlines):
            input_headline_idx = self.sentence2idx(each_headline, True, max_len_head, True)
            predicated_headline_idx = self.sentence2idx(each_headline, True, max_len_head, False)
            desc_idx = self.sentence2idx(each_desc, False, max_len_desc)

            # assert size checks
            assert len(input_headline_idx) == max_len_head - 1
            assert len(predicated_headline_idx) == max_len_head
            assert len(desc_idx) == max_len_desc + 1

            X.append(desc_idx + input_headline_idx)
            y.append(predicated_headline_idx)
        
        X, y = np.array(X), np.array(y)
        if is_training:
            X = self.flip_words_randomly(X, number_words_to_replace, model)
            # One hot encoding of y
            vocab_size = self.word2vec.shape[0]
            length_of_data = len(headlines)
            Y = np.zeros((length_of_data, max_len_head, vocab_size))
            for i, each_y in enumerate(y):
                Y[i, :, :] = np_utils.to_categorical(each_y, vocab_size)
            #check equal lengths
            assert len(X)==len(Y)
            return X, Y
        else:
            #Testing doesnot require OHE form of headline, flipping also not required
            #Because BLUE score require words and not OHE form to check accuracy
            return X,headlines
        
    def read_small_data_files(self, file_name='../../temp_results/raw_news_text.txt', seperator='#|#'):
        """
        Assumes one line contatin "headline seperator description"
        """
        X, y = [], []
        with codecs.open(file_name,'r',encoding='utf8') as fp:
            for each_line in fp:
                each_line = each_line.strip()
                headline, desc = each_line.split(seperator)
                input_headline_idx = self.sentence2idx(headline, True, max_len_head, True)
                predicated_headline_idx = self.sentence2idx(headline, True, max_len_head, False)
                desc_idx = self.sentence2idx(desc, False, max_len_desc)
                # assert size checks
                assert len(input_headline_idx) == max_len_head - 1
                assert len(predicated_headline_idx) == max_len_head
                assert len(desc_idx) == max_len_desc + 1

                X.append(desc_idx + input_headline_idx)
                y.append(predicated_headline_idx)

        return (X, y)
    
    def shuffle_file(self,file_name):
        try:
            subprocess.check_output(['shuf',file_name,"--output="+file_name])
            print ("Input file shuffled!")
        except:
            print ("Input file NOT shuffled as shuf command not available!")

    def large_file_reading_generator(self, file_name):
        """
        read large file line by line
        """
        while True:
            with codecs.open(file_name,'r',encoding='utf8') as file_pointer:
                for each_line in file_pointer:
                    yield each_line.strip()
            self.shuffle_file(file_name)

    def data_generator(self, file_name, batch_size, number_words_to_replace, model, seperator='#|#',is_training=True):
        """
        read large file in chunks and return chunk of data to train on
        """
        file_iterator = self.large_file_reading_generator(file_name)
        while True:
            X, y = [], []
            for i in xrange(batch_size):
                each_line = next(file_iterator)
                headline, desc = each_line.split(seperator)
                X.append(desc)
                y.append(headline)
            yield self.convert_inputs(X, y, number_words_to_replace, model,is_training)

    def OHE_to_indexes(self,y_val):
        """
        reverse of OHE 
        OHE => indexes
        e.g. [[0,0,1],[1,0,0]] => [2,0]
        """
        list_of_headline = []
        for each_headline in y_val:
            list_of_word_indexes = np.where(np.array(each_headline)==1)[1]
            list_of_headline.append(list(list_of_word_indexes))
        return list_of_headline
    
    def indexes_to_words(self,list_of_headline):
        """
        indexes => words (for BLUE Score)
        e.g. [2,0] => ["I","am"] (idx2word defined dictionary used)
        """
        list_of_word_headline = []
        for each_headline in list_of_headline:
            each_headline_words = []
            for each_word in each_headline:
                #Dont include <eos> and <empty> tags
                if each_word in (empty_tag_location, eos_tag_location, unknown_tag_location):
                    continue
                each_headline_words.append(self.idx2word[each_word])
            list_of_word_headline.append(each_headline_words)            
        return list_of_word_headline
    
    def blue_score_text(self,y_actual,y_predicated):
        #check length equal
        assert len(y_actual) ==  len(y_predicated)
        #list of healine .. each headline has words
        no_of_news = len(y_actual)
        blue_score = 0.0
        for i in range(no_of_news):
            reference = y_actual[i]
            hypothesis = y_predicated[i]
            
            #Avoid ZeroDivisionError in blue score
            #default weights
            weights=(0.25, 0.25, 0.25, 0.25)
            min_len_present = min(len(reference),len(hypothesis))
            if min_len_present==0:
                continue
            if min_len_present<4:
                weights=[1.0/min_len_present,]*min_len_present
            
            blue_score = blue_score + sentence_bleu([reference],hypothesis,weights=weights)
        
        return blue_score/float(no_of_news)


    def blue_score_calculator(self, model, validation_file_name, no_of_validation_sample, validation_step_size):
        #In validation don't repalce with random words
        number_words_to_replace=0
        temp_gen = self.data_generator(validation_file_name, validation_step_size, number_words_to_replace, model)        
        
        total_blue_score = 0.0            
        blue_batches = 0
        blue_number_of_batches = no_of_validation_sample / validation_step_size
        for X_val, y_val in temp_gen:
            y_predicated = model.predict_classes(X_val,batch_size=validation_step_size)
            y_predicated_words = self.indexes_to_words(y_predicated)
            list_of_word_headline = self.indexes_to_words(self.OHE_to_indexes(y_val))
            assert len(y_val)==len(list_of_word_headline) 

            total_blue_score = total_blue_score + self.blue_score_text(list_of_word_headline, y_predicated_words)
            
            blue_batches += 1
            if blue_batches >=  blue_number_of_batches:
                #get out of infinite loop of val generator
                break
            if blue_batches%10==0:
                print ("eval for {} out of {}".format(blue_batches, blue_number_of_batches))

        #close files and delete generator  
        del temp_gen
        return total_blue_score/float(blue_batches)          

    def train(self, model, data_file_name, validation_file_name, no_of_training_sample, train_batch_size, no_of_validation_sample, validation_step_size, no_of_epochs, number_words_to_replace,model_weights_file_name):
        """
        trains a model
        Manually loop (without using internal epoch parameter of keras),
        train model for each epoch, evaluate logloss and BLUE score of model on validation data
        save model if BLUE/logloss score improvement ...
        save score history for plotting purposes.
        Note : validation step size meaning over here is different from keras
        here validation_step_size means, batch_size in which blue score evaluated
        after all batches processed, blue scores over all batches averaged to get one blue score.
        """
        #load model weights if file present 
        if os.path.isfile(model_weights_file_name):
            print ("loading weights already present in {}".format(model_weights_file_name))
            model.load_weights(model_weights_file_name)
            print ("model weights loaded for further training")
            
        data_generator = self.data_generator(data_file_name, train_batch_size, number_words_to_replace, model)
        
        blue_scores = []
        #blue score are always greater than 0
        best_blue_score_track = -1.0
        number_of_batches = math.ceil(no_of_training_sample / float(train_batch_size))
        
        for each_epoch in range(no_of_epochs):
            print ("running for epoch ",each_epoch)
            start_time = time.time()
            
            #manually loop over batches and feed to network
            #purposefully not used fit_generator
            batches = 0
            for X_batch, Y_batch in data_generator:
                model.fit(X_batch,Y_batch,batch_size=train_batch_size,epochs=1)
                batches += 1
                #take last chunk and roll over to start ...
                #therefore float used ... 
                if batches >= number_of_batches :
                    break
                if batches%10==0:
                    print ("training for {} out of {} for epoch {}".format(batches, number_of_batches, each_epoch))
                    
            end_time = time.time()
            print("time to train epoch ",end_time-start_time)

            # evaluate model on BLUE score and save best BLUE score model...
            blue_score_now = self.blue_score_calculator(model,validation_file_name,no_of_validation_sample,validation_step_size)
            blue_scores.append(blue_score_now)
            if best_blue_score_track < blue_score_now:
                best_blue_score_track = blue_score_now
                print ("saving model for blue score ",best_blue_score_track)
                model.save_weights(model_weights_file_name)
                
            # Note : It saves on every loop, this looks REPETATIVE, BUT
            # if user aborts(control-c) in middle of epochs then we get previous
            # present history
            # User can track previous history while model running ... 
            # dump history object list for further plotting of loss
            # append BLUE Score for to another list  and dump for futher plotting
            with open("../../temp_results/blue_scores.pickle", "wb") as output_file:
                pickle.dump(blue_scores, output_file)
    
    def is_headline_end(self, word_index_list, current_predication_position):
        """
        is headline ended checker
        current_predication_position is 0 index based
        """
        if (word_index_list is None) or (len(word_index_list)==0):
            return False
        if word_index_list[current_predication_position]==eos_tag_location or current_predication_position>=max_length:
            return True
        return False
        
    def process_word(self, predication, word_position_index, top_k, X, prev_layer_log_prob):
        """
        Extract top k predications of given position
        """
        #predication conttains only one element
        #shape of predication (1,max_head_line_words,vocab_size)
        predication = predication[0]
        #predication (max_head_line_words,vocab_size)
        predication_at_word_index = predication[word_position_index]
        #http://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        sorted_arg = predication_at_word_index.argsort()
        top_probable_indexes = sorted_arg[-top_k:][::-1]
        top_probabilities = np.take(predication_at_word_index,top_probable_indexes)
        log_probabilities = np.log(top_probabilities)
        #make sure elements doesnot contain -infinity
        log_probabilities[log_probabilities == -inf] = -sys.maxint - 1
        #add prev layer probability
        log_probabilities = log_probabilities + prev_layer_log_prob
        assert len(log_probabilities)==len(top_probable_indexes)
        
        #add previous words ... preparation for next input
        #offset calculate ... description + eos + headline till now
        offset = max_len_desc+word_position_index+1
        ans = []
        for i,j in zip(log_probabilities, top_probable_indexes):
            next_input = np.concatenate((X[:offset], [j,]))
            next_input = next_input.reshape((1,next_input.shape[0]))
            #for the last time last word put at max_length + 1 position 
            #don't truncate that
            if offset!=max_length:
                next_input = sequence.pad_sequences(next_input, maxlen=max_length, value=empty_tag_location, padding='post', truncating='post')
            next_input = next_input[0]
            ans.append((i,next_input))
        #[(prob,list_of_words_as_next_input),(prob2,list_of_words_as_next_input2),...]
        return ans
        
    def beam_search(self,model,X,top_k):
        """
        1.Loop over max headline word allowed
        2.predict word prob and select top k words for each position
        3.select top probable combination uptil now for next round
        """
        #contains [(log_p untill now, word_seq), (log_p2, word_seq2)]
        prev_word_index_top_k = []
        curr_word_index_top_k = []
        done_with_pred = []
        #1d => 2d array [1,2,3] => [[1,2,3]]
        data = X.reshape((1,X.shape[0]))
        #shape of predication (1,max_head_line_words,vocab_size)
        predication = model.predict_proba(data,verbose=0)
        #prev layer probability 1 => np.log(0)=0.0
        prev_word_index_top_k = self.process_word(predication,0,top_k,X,0.0)
        
        #1st time its done above to fill prev word therefore started from 1
        for i in range(1,max_len_head):
            #i = represents current intrested layer ...
            for j in range(len(prev_word_index_top_k)):
                #j = each time loops for top k results ...
                probability_now, current_intput = prev_word_index_top_k[j]
                data = current_intput.reshape((1,current_intput.shape[0]))
                predication = model.predict_proba(data,verbose=0)
                next_top_k_for_curr_word = self.process_word(predication,i,top_k,current_intput,probability_now)
                curr_word_index_top_k = curr_word_index_top_k + next_top_k_for_curr_word
                
            #sort new list, empty old, copy top k element to old, empty new
            curr_word_index_top_k = sorted(curr_word_index_top_k,key=itemgetter(0),reverse=True)
            prev_word_index_top_k_temp = curr_word_index_top_k[:top_k]
            curr_word_index_top_k = []
            prev_word_index_top_k = []
            #if word predication eos ... put it done list ...
            for each_proba, each_word_idx_list in prev_word_index_top_k_temp:
                offset = max_len_desc+i+1
                if self.is_headline_end(each_word_idx_list,offset):
                    done_with_pred.append((each_proba, each_word_idx_list))
                else:
                    prev_word_index_top_k.append((each_proba,each_word_idx_list))
            
        #sort according to most probable
        done_with_pred = sorted(done_with_pred,key=itemgetter(0),reverse=True)
        done_with_pred = done_with_pred[:top_k]
        return done_with_pred
            
    def test(self, model, data_file_name, no_of_testing_sample, model_weights_file_name,top_k,output_file,seperator='#|#'):
        """
        test on given description data file with empty headline ...
        """
        model.load_weights(model_weights_file_name)
        print ("model weights loaded")
        #Always 1 for now ... later batch code for test sample created
        test_batch_size = 1
        data_generator = self.data_generator(data_file_name, test_batch_size, number_words_to_replace=0, model=None,is_training=False)
        number_of_batches = math.ceil(no_of_testing_sample / float(test_batch_size))
        
        with codecs.open(output_file, 'w',encoding='utf8') as f:
            #testing batches
            batches = 0
            for X_batch, Y_batch in data_generator:
                #Always come one becaise X_batch contains one element
                X = X_batch[0]
                Y = Y_batch[0]
                assert X[max_len_desc]==eos_tag_location
                #wipe up news headlines present and replace by empty tag ...            
                X[max_len_desc+1:]=empty_tag_location
                result = self.beam_search(model,X,top_k)
                #take top most probable element
                list_of_word_indexes = result[0][1]
                list_of_words = self.indexes_to_words([list_of_word_indexes])[0]
                headline = u" ".join(list_of_words[max_len_head+1:])
                f.write(Y+seperator+headline+"\n")
                batches += 1
                #take last chunk and roll over to start ...
                #therefore float used ... 
                if batches >= number_of_batches :
                    break
                if batches%10==0:
                    print ("working on batch no {} out of {}".format(batches,number_of_batches))
                
    def pre_process(self,top_k,file_names,word2vec_file_name,is_train):
        p = preprocess()
        new_file_name = p.new_file_name(word2vec_file_name,top_k)
        #if training create new word embedding file of top k words
        if is_train:
            print("creating new word2vec file of top {} fetaures".format(top_k))
            top_k_words = p.top_k_freq_words(file_names,top_k)        
            p.top_k_word2vec(word2vec_file_name,top_k_words,embedding_dimension,new_file_name)
        print("loading word2vec file from ",new_file_name)
        self.read_word_embedding(new_file_name)
    
if __name__ == '__main__':
    
    data_file_name='../../temp_results/train_corpus.txt'
    validation_file_name='../../temp_results/validation_corpus.txt'
    test_file_name='../../temp_results/test_corpus.txt'
    word_embedding_file_name = '../../temp_results/word2vec_hindi.txt'
    model_weights_file_name = '../../temp_results/deep_news_model_weights.h5'
    output_file='../../temp_results/test_output.txt'
    all_file_names = [data_file_name,validation_file_name,test_file_name]
    is_train = True
    
    t = news_rnn()
    t.pre_process(top_freq_word_to_use,all_file_names,
                                             word_embedding_file_name,is_train)
    model = t.create_model()
    if is_train:
        t.train(model=model, 
                data_file_name=data_file_name, 
                validation_file_name=validation_file_name, 
                no_of_training_sample=t.file_line_counter(data_file_name), 
                train_batch_size=32,
                no_of_validation_sample=t.file_line_counter(validation_file_name),
                validation_step_size=32, 
                no_of_epochs=16, 
                number_words_to_replace=2,
                model_weights_file_name=model_weights_file_name)
    else:
        t.test(model=model,
               data_file_name=test_file_name,
               no_of_testing_sample=t.file_line_counter(test_file_name),
               model_weights_file_name=model_weights_file_name,
               top_k=10,
               output_file=output_file) 

        