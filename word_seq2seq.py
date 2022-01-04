# Used source(s): https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

from __future__ import print_function
from gensim.utils import simple_preprocess
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import pickle

batch_size = 64  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 19000  # Number of samples to train on.


class Seq2seqModel:
    """"The translation model"""
    def __init__(self):
        self.input_texts = []
        self.target_texts = []
        self.input_token_index = dict()
        self.target_token_index = dict()
        self.encoder_vocab_size = None
        self.decoder_vocab_size = None
        self.max_encoder_seq_length = None
        self.max_decoder_seq_length = None
        self.model = None
        self.model_type = None

        self.decoder_input_data = None
        self.decoder_target_data = None
        self.encoder_input_data = None

        self.encoder_inputs = None
        self.encoder_states = None
        self.decoder_inputs = None
        self.decoder_outputs = None
        self.decoder_dense = None
        self.decoder_network = None
        self.input_tokenizer = None
        self.target_tokenizer = None

    def load_all(self, model_name, data_path, model_type, optimizer, reverse):
        """Loads existing tokenizers, sentence pairs and translation model

        :param model_name: name of the file the model was saved into
        :param data_path: the path to the sentence pairs
        :param model_type: type of the model, gru/lstm
        :param optimizer: name of the optimizer the model uses
        :param reverse: true if the model uses the sentence pairs data in the opposite order
        """
        with open('./saved_models/word_seq2seq/' + model_name + '_input_tokenizer.pickle', 'rb') as handle:
            self.input_tokenizer = pickle.load(handle)

        with open('./saved_models/word_seq2seq/' + model_name + '_target_tokenizer.pickle', 'rb') as handle:
            self.target_tokenizer = pickle.load(handle)

        self.load_data(data_path, reverse=reverse)
        self.create_model_input(model_type)
        self.load_model(model_name, optimizer)

    def load_data(self, data_path, reverse=False):
        """Loads the sentence pairs

        :param data_path: the path to the sentence pairs
        :param reverse: true if the model uses the sentence pairs data in the opposite order
        """
        # read the translation sentence pairs
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            if reverse:
                target_text, input_text = line.split('\t')
            else:
                input_text, target_text = line.split('\t')

            # startsequence and endsequence are unique words for start and end token substitution
            target_text = 'startsequence ' + target_text + ' endsequence'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)

        if self.input_tokenizer is None and self.target_tokenizer is None:
            self.input_tokenizer = Tokenizer(oov_token=1)
            self.target_tokenizer = Tokenizer(oov_token=1)
            self.input_tokenizer.fit_on_texts(self.input_texts)
            self.target_tokenizer.fit_on_texts(self.target_texts)

        # retrieve vocabulary size
        self.encoder_vocab_size = len(self.input_tokenizer.word_index) + 1
        self.decoder_vocab_size = len(self.target_tokenizer.word_index) + 1

        self.max_encoder_seq_length = max([len(self.input_tokenizer.texts_to_sequences([txt])[0]) for txt
                                           in self.input_texts])
        self.max_decoder_seq_length = max([len(self.target_tokenizer.texts_to_sequences([txt])[0]) for txt
                                           in self.target_texts])

        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', self.encoder_vocab_size)
        print('Number of unique output tokens:', self.decoder_vocab_size)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        self.create_dict()

    def create_dict(self):
        """Saves the tokenizers' dictionaries to separate variables"""
        self.input_token_index = self.input_tokenizer.word_index
        self.target_token_index = self.target_tokenizer.word_index

    def create_model_input(self, model_type):
        """Create the input/output tensors for the seq2seq model and calls the function making the model

        :param model_type: type of the model, gru/lstm
        """

        sample_count = len(self.input_texts)
        self.encoder_input_data = np.zeros((sample_count, self.max_encoder_seq_length, self.encoder_vocab_size),
                                           dtype='float32')
        self.decoder_input_data = np.zeros((sample_count, self.max_decoder_seq_length, self.decoder_vocab_size),
                                           dtype='float32')
        self.decoder_target_data = np.zeros((sample_count, self.max_decoder_seq_length, self.decoder_vocab_size),
                                            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for j, token in enumerate(self.input_tokenizer.texts_to_sequences([input_text])[0]):
                self.encoder_input_data[i, j, token] = 1.

            for j, token in enumerate(self.target_tokenizer.texts_to_sequences([target_text])[0]):
                self.decoder_input_data[i, j, token] = 1.
                if j > 0:
                    self.decoder_target_data[i, j - 1, token] = 1.

        if model_type == 'gru':
            self.model_type = model_type
            self.gru()
        elif model_type == 'lstm':
            self.model_type = model_type
            self.lstm()
        else:
            print('Unknown model type.')

    def gru(self):
        """Makes a gru based seq2seq model"""
        self.encoder_inputs = Input(shape=(None, self.encoder_vocab_size))
        encoder_outputs, self.encoder_states = GRU(latent_dim, return_state=True, dropout=0.1)(self.encoder_inputs)

        self.decoder_inputs = Input(shape=(None, self.decoder_vocab_size))
        self.decoder_network = GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.1)
        self.decoder_outputs, _ = self.decoder_network(self.decoder_inputs, initial_state=self.encoder_states)

        self.decoder_dense = Dense(self.decoder_vocab_size, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

    def lstm(self):
        """Makes an lstm based seq2seq model"""
        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, self.encoder_vocab_size))
        encoder = LSTM(latent_dim, return_state=True, dropout=0.1)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_inputs = Input(shape=(None, self.decoder_vocab_size))

        self.decoder_network = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.1)
        self.decoder_outputs, _, _ = self.decoder_network(self.decoder_inputs, initial_state=self.encoder_states)

        self.decoder_dense = Dense(self.decoder_vocab_size, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

    def train_model(self, opt):
        """Trains a model with on the already loaded data

        :param opt: choice of optimizer
        """
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)

        # Run training
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        self.model.compile(optimizer=opt, loss='categorical_crossentropy')
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                       batch_size=batch_size, verbose=2,
                       epochs=epochs,
                       validation_split=0.15, callbacks=[early_stop])

    def load_model(self, model_name, opt):
        """Loads the translation model

        :param model_name: name of the file the model was saved into
        :param opt: name of the optimizer the model uses
        """
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        self.model.load_weights('./saved_models/word_seq2seq/' + model_name + '.h5')
        self.model.compile(optimizer=opt, loss='categorical_crossentropy')

    def save_model(self, model_name):
        """Saves the translation model and tokenizers

        :param model_name: name of the model that will be used in the filenames
        """
        self.model.save_weights('./saved_models/word_seq2seq/' + model_name + ".h5")
        print("Saved model to disk")

        with open('./saved_models/word_seq2seq/' + model_name + '_input_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.input_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./saved_models/word_seq2seq/' + model_name + '_target_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.target_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def decode_sequence(self, input_sentence):
        """Inference model for translating a sentence

        :param input_sentence: the sentence to be translated
        :returns the offered translation
        """
        input_seq = np.zeros((1, self.max_encoder_seq_length, self.encoder_vocab_size), dtype='float32')

        for j, token in enumerate(self.input_tokenizer.texts_to_sequences([input_sentence])[0]):
            input_seq[0, j, token] = 1.

        encoder_model = Model(self.encoder_inputs, self.encoder_states)

        if self.model_type == 'lstm':
            decoder_state_input_h = Input(shape=(latent_dim,))
            decoder_state_input_c = Input(shape=(latent_dim,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs, state_h, state_c = self.decoder_network(self.decoder_inputs,
                                                                     initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
        else:
            decoder_states_inputs = [Input(shape=(latent_dim,))]
            decoder_outputs, state = self.decoder_network(self.decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state]

        decoder_outputs = self.decoder_dense(decoder_outputs)
        decoder_model = Model([self.decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())

        # encode the input as state vectors
        states_value = encoder_model.predict(input_seq)

        # generate empty target sequence of length 1
        target_seq = np.zeros((1, 1, self.decoder_vocab_size))
        # populate the first character of target sequence with the start character
        target_seq[0, 0, self.target_token_index['startsequence']] = 1.

        # sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1)
        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            if self.model_type == 'lstm':
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
                states_value = [h, c]
            else:
                output_tokens, states = decoder_model.predict([target_seq] + [states_value])
                states_value = states

            # sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = reverse_target_char_index[sampled_token_index]
            if sampled_word == 'endsequence':
                return decoded_sentence
            else:
                decoded_sentence += sampled_word + ' '

            # exit condition: either hit max length or find stop character
            if len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            # update the target sequence (of length 1)
            target_seq = np.zeros((1, 1, self.decoder_vocab_size))
            target_seq[0, 0, sampled_token_index] = 1.

        return decoded_sentence
    
    def test_model(self, test_data):
        """Evaluates on test sentences with BLEU score computation

        :param test_data: the test sentences
        :returns the average of the BLEU scores
        """
        score_sum = 0
        cc = SmoothingFunction()
        for line in test_data:
            sentence, translation = line.split('\t')
            model_translation = self.decode_sequence(sentence)
            score = sentence_bleu([simple_preprocess(translation, min_len=1)],
                                  simple_preprocess(model_translation, min_len=1), smoothing_function=cc.method4)
            print(translation+'\t'+model_translation+'\t'+str(score))
            score_sum = score_sum + score

        average_score = score_sum / len(test_data)
        print('Average score: ' + str(average_score))

        return average_score


def execute(network_type, opt='rmsprop', reverse=False):
    model = Seq2seqModel()
    model.load_data('./datasets/danish-english_train.txt', reverse=reverse)
    model.create_model_input(network_type)
    model.train_model(opt=opt)
    if reverse:
        lang = '_english_danish'
    else:
        lang = '_danish_english'
    model.save_model(network_type + '_' + str(batch_size) + '_' + str(latent_dim) + lang)


def load(batch, latent, network):
    model = Seq2seqModel()
    model.load_all(str(network) + '_' + str(batch) + '_' + str(latent)
                   + '_danish_english', './datasets/danish-english_train.txt', network, 'rmsprop', True)

    with open('./datasets/danish-english_test.txt') as f:
        lines = f.read().splitlines()
    model.test_model(lines)
