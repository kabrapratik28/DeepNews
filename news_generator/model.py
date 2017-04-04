import random
import cPickle as pickle
import numpy as np


import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
from keras.utils import np_utils

from tqdm import tqdm
from sklearn.cross_validation import train_test_split
from nltk.translate.bleu_score import sentence_bleu

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

# number of validation examples
nb_val_samples = 3000
embedding_dimension = 300
max_len_head = 25
max_len_desc = 25
max_length = max_len_head + max_len_desc
rnn_layers = 3
rnn_size = 512
# first 40 numebers from hidden layer output used for
# simple context calculation
activation_rnn_size = 40
empty_tag_location = 0
eos_tag_location = 1
learning_rate = 1e-4

class news_rnn(object):
    def __init__(self,):
        self.word2vec = None
        self.idx2word = {}
        self.word2idx = {}

        # initalize end of sentence and empty
        self.word2idx['<empty>'] = empty_tag_location
        self.word2idx['<eos>'] = eos_tag_location
        self.idx2word[empty_tag_location] = '<empty>'
        self.idx2word[eos_tag_location] = '<eos>'

        self.cache_validation_data = None
        # TODO: make model as part of self.model
        # TODO: store/load this dictionaries from pickle
    
    def file_line_counter(self,file_name):
        with open(file_name) as f:
            for i, l in tqdm(enumerate(f)):
                pass
        return i+1
    
    def read_word_embedding(self, file_name='../../temp_results/word2vec_hindi.txt'):
        """
        read word embedding file and assign indexes to word
        """
        idx = 2
        temp_word2vec_dict = {}
        # TODO: <empty>, <eos> vectors initializations?
        # <empty>, <eos> tag replaced by word2vec learning
        # create random dimensional vector for empty and eos
        temp_word2vec_dict['<empty>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]
        temp_word2vec_dict['<eos>'] = [float(i) for i in np.random.rand(embedding_dimension, 1)]

        with open(file_name) as fp:
            for each_line in fp:
                word_embedding_data = each_line.split(" ")
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
                count = count + 1
                if count >= curr_max_length:
                    break

        if is_headline:
            return self.headline2idx(list_idx, curr_max_length, is_input)
        else:
            return self.desc2idx(list_idx, curr_max_length)

    def read_small_data_files(self, file_name='../../temp_results/raw_news_text.txt', seperator='#|#'):
        """
        Assumes one line contatin "headline seperator description"
        """
        X, y = [], []
        with open(file_name) as fp:
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

        input_shape[0] = batch_szie remains as it is
        maxlenh =
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
                # dont replace by empty location
                if word_idx == empty_tag_location:
                    continue
                copy_data[idx, replace_idx] = word_idx
        return copy_data

    def convert_inputs(self, descriptions, headlines, number_words_to_replace, model):
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

    def large_file_reading_generator(self, file_name):
        """
        read large file line by line
        """
        while True:
            with open(file_name) as file_pointer:
                for each_line in file_pointer:
                    yield each_line.strip()
            # TODO: shuffle file lines for next epoch

    def data_generator(self, file_name, batch_size, number_words_to_replace, model, seperator='#|#'):
        """
        read large file in chunks and return chunk of data to train on
        """
        file_iterator = iter(self.large_file_reading_generator(file_name))
        while True:
            X, y = [], []
            for i in xrange(batch_size):
                each_line = next(file_iterator)
                desc, headline = each_line.split(seperator)
                X.append(desc)
                y.append(headline)
            yield self.convert_inputs(X, y, number_words_to_replace, model)

    def OHE_to_indexes(self,y_val):
        """
        reverse of OHE 
        OHE => indexes
        e.g. [[0,0,1],[1,0,0]] => [2,0]
        """
        print ("Converting OHE to indexes")
        list_of_headline = []
        for each_headline in tqdm(y_val):
            list_of_word_indexes = np.where(np.array(each_headline)==1)[1]
            list_of_headline.append(list(list_of_word_indexes))
        return list_of_headline
    
    def indexes_to_words(self,list_of_headline):
        """
        indexes => words (for BLUE Score)
        e.g. [2,0] => ["I","am"] (idx2word defined dictionary used)
        """
        print ("Converting indexes to words")
        list_of_word_headline = []
        for each_headline in tqdm(list_of_headline):
            each_headline_words = []
            for each_word in each_headline:
                #Dont include <eos> and <empty> tags
                if each_word==empty_tag_location or each_word==eos_tag_location:
                    continue
                each_headline_words.append(self.idx2word[each_word])
            list_of_word_headline.append(each_headline_words)            
        return list_of_word_headline

    def blue_score_calculator(self, model,validation_file_name,validation_step_size, number_words_to_replace):
        if self.cache_validation_data == None:
            print ("validation data creating ...")
            #In validation don't repalce with random words
            number_words_to_replace=0
            temp_gen = self.data_generator(validation_file_name, validation_step_size, number_words_to_replace, model)        
            print("caching validation data ... ")
            self.cache_validation_data = {}
            for X_val, y_val in temp_gen:
                self.cache_validation_data["X_val"] = X_val
                self.cache_validation_data["y_val"] = y_val
                
                list_of_word_headline = self.indexes_to_words(self.OHE_to_indexes(y_val))
                self.cache_validation_data["word_headlines"]=list_of_word_headline
                assert len(y_val)==len(list_of_word_headline)    
                #get out of infinite loop of val generator
                break

            #close files and delete generator  
            del temp_gen
            
        #Found cache ... 
        #100 examples processed each batch ... 
        y_predicated = model.predict_classes(self.cache_validation_data["X_val"],batch_size=100)
        #list of healine .. each headline has words
        y_predicated_words = self.indexes_to_words(y_predicated)
        assert len(self.cache_validation_data["word_headlines"]) ==  len(y_predicated_words)
        no_of_news = len(self.cache_validation_data["word_headlines"])
        blue_score = 0.0
        for i in range(no_of_news):
            reference = self.cache_validation_data["word_headlines"][i]
            hypothesis = y_predicated_words[i]
            
            #Avoid ZeroDivisionError in blue score
            #default weights
            weights=(0.25, 0.25, 0.25, 0.25)
            min_len_present = min(len(reference),len(hypothesis))
            if min_len_present==0:
                continue
            if min_len_present<4:
                weights=[1.0/min_len_present,]*min_len_present
            
            blue_score = blue_score + sentence_bleu([reference],hypothesis,weights=weights)
        
        return blue_score/no_of_news

    def train(self, model, data_file_name, validation_file_name, no_of_training_sample, train_batch_size, validation_step_size, no_of_epochs, number_words_to_replace):
        """
        trains a model
        Manually loop (without using internal epoch parameter of keras),
        train model for each epoch, evaluate logloss and BLUE score of model on validation data
        save model if BLUE/logloss score improvement ...
        save score history for plotting purposes.
        """
        data_generator = self.data_generator(data_file_name, train_batch_size, number_words_to_replace, model)
        validation_generator = self.data_generator(validation_file_name, validation_step_size, number_words_to_replace, model)
        
        histories = []
        blue_scores = []
        #blue score are always greater than 0
        best_blue_score_track = -1.0
        for each_epoch in range(no_of_epochs):
            print ("running for epoch ",each_epoch)
            history = model.fit_generator(
                        data_generator,
                        steps_per_epoch=no_of_training_sample / train_batch_size,
                        epochs=1,
                        validation_data=validation_generator, validation_steps=validation_step_size
                    )
            histories.append(history)

            # evaluate model on BLUE score and save best BLUE score model...
            blue_score_now = self.blue_score_calculator(model,validation_file_name,validation_step_size)
            blue_scores.append(blue_score_now)
            if best_blue_score_track < blue_score_now:
                best_blue_score_track = blue_score_now
                print ("saving model for blue score ",best_blue_score_track)
                model.save('../../temp_results/deep_news_model_weights.h5')
                
            # Note : It saves on every loop, this looks REPETATIVE, BUT
            # if user aborts(control-c) in middle of epochs then we get previous
            # present history
            # User can track previous history while model running ... 
            # dump history object list for further plotting of loss
            # append BLUE Score for to another list  and dump for futher plotting
            with open("../../temp_results/histories.pickle", "wb") as output_file:
                pickle.dump(histories, output_file)
            with open("../../temp_results/blue_scores.pickle", "wb") as output_file:
                pickle.dump(blue_scores, output_file)

if __name__ == '__main__':
    
    data_file_name='../../temp_results/train_corpus.txt'
    validation_file_name='../../temp_results/validation_corpus.txt'
    
    t = news_rnn()
    t.read_word_embedding()
    model = t.create_model()
    t.train(model=model, 
            data_file_name=data_file_name, 
            validation_file_name=validation_file_name, 
            no_of_training_sample=t.file_line_counter(data_file_name), 
            train_batch_size=500, 
            validation_step_size=t.file_line_counter(validation_file_name), 
            no_of_epochs=28, 
            number_words_to_replace=2)