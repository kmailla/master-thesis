# Used source(s): https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
import numpy as np
from keras.callbacks import EarlyStopping

batch_size = 32  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 100000  # Number of samples to train on.


class Seq2seqModel:
    """The spelling correction model"""
    def __init__(self):
        self.input_texts = []
        self.target_texts = []
        self.input_token_index = dict()
        self.target_token_index = dict()
        self.num_encoder_tokens = None
        self.num_decoder_tokens = None
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
        
    def load_all(self, model_name, data_path, model_type):
        """Loads existing sentence pairs and translation model

        :param model_name: name of the file the model was saved into
        :param data_path: the path to the sentence pairs
        :param model_type: type of the model, gru/lstm
        :returns the loaded model
        """
        self.load_data(data_path)
        self.create_model_input(model_type)
        self.load_model(model_name)

        return self.model

    def load_data(self, data_path):
        """Loads the sentence pairs

        :param data_path: the path to the sentence pairs
        """
        input_characters = set()
        target_characters = set()
        # Vectorize the data.
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        # create dictionary
        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

    def create_model_input(self, model_type):
        """Create the input/output tensors for the seq2seq model and calls the function making the model

        :param model_type: type of the model, gru/lstm
        """
        sample_count = len(self.input_texts)
        self.encoder_input_data = np.zeros((sample_count, self.max_encoder_seq_length,
                                            self.num_encoder_tokens), dtype='float32')
        self.decoder_input_data = np.zeros((sample_count, self.max_decoder_seq_length,
                                            self.num_decoder_tokens), dtype='float32')
        self.decoder_target_data = np.zeros((sample_count, self.max_decoder_seq_length,
                                             self.num_decoder_tokens), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
        
        if model_type == 'gru':
            self.gru()
            self.model_type = 'gru'
        elif model_type == 'lstm':
            self.lstm()
            self.model_type = 'lstm'
        else:
            print('Unknown model type.')

    def gru(self):
        """Makes a gru based seq2seq model"""
        self.encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = GRU(latent_dim, return_state=True, dropout=0.1)
        encoder_outputs, self.encoder_states = encoder(self.encoder_inputs)

        self.decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        self.decoder_network = GRU(latent_dim, return_sequences=True, return_state=True, dropout=0.1)
        self.decoder_outputs, _ = self.decoder_network(self.decoder_inputs, initial_state=self.encoder_states)

        self.decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

    def lstm(self):
        """Makes an lstm based seq2seq model"""
        # Define an input sequence and process it.
        self.encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True, dropout=0.1)
        encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        self.encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        self.decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_network = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.1)
        self.decoder_outputs, _, _ = self.decoder_network(self.decoder_inputs, initial_state=self.encoder_states)

        self.decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

    def train_model(self):
        """Trains a model with on the already loaded data"""
        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)

        # Run training
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=0.15, verbose=2, callbacks=[early_stop])
        print(self.model.summary())

    def load_model(self, model_name):
        """Loads the translation model

        :param model_name: name of the file the model was saved into
        :returns the loaded model
        """
        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        self.model.load_weights('./saved_models/char_seq2seq/' + model_name + '.h5')
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        return self.model

    def save_model(self, model_name):
        """Saves the translation model and tokenizers

        :param model_name: name of the model that will be used in the filenames
        """
        self.model.save_weights('./saved_models/char_seq2seq/' + model_name + ".h5")
        print("Saved model to disk")

    def correct_sequence(self, input_sentence):
        """Inference model for translating a sentence

        :param input_sentence: the sentence to be translated
        :returns the offered word correction
        """
        input_seq = np.zeros((1, self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')

        for t, char in enumerate(input_sentence):
            input_seq[0, t, self.input_token_index[char]] = 1.

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

        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            if self.model_type == 'lstm':
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
                states_value = [h, c]
            else:
                output_tokens, states = decoder_model.predict([target_seq] + [states_value])
                states_value = states

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length:
                return decoded_sentence

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            decoded_sentence += sampled_char

        return decoded_sentence

    def test_model(self, test_data, dictionary):
        """Evaluates on test misspellings with equality check and dictionary lookup

        :param test_data: the test misspellings
        :param dictionary: a dictionary that should contain all the right words on a given language
        """
        all_pairs = len(test_data)
        exact_matches = 0
        dict_words = 0

        for row in test_data:
            is_exact_match = 0
            is_in_dict = 0

            bad_word, correct_word = row.split('\t')
            correction = self.correct_sequence(bad_word)
            if correction == correct_word:
                exact_matches += 1
                is_exact_match = 1
            if correction in dictionary:
                dict_words += 1
                is_in_dict = 1
            
            pline = bad_word+'\t'+correct_word+'\t'+correction+'\t'+str(is_exact_match)+'\t'+str(is_in_dict)
            print(pline)

        print('exact corrections: ', float(exact_matches)/float(all_pairs))
        print('in-dictionary corrections: ', float(dict_words) / float(all_pairs))


def execute(network_type):
    model = Seq2seqModel()
    model.load_data('./datasets/english_misspellings_train.txt')
    model.create_model_input(network_type)
    model.train_model()
    model.save_model('misspelling_'+network_type+'_'+str(latent_dim)+'_'+str(batch_size)+'_eng')


def load(network_type, units, batch):
    model = Seq2seqModel()
    model.load_all('misspelling_'+network_type+'_'+str(units)+'_'+str(batch)+'_eng',
                   './datasets/english_misspellings_train.txt', network_type)

    print(model.correct_sequence('kat'))
    print(model.correct_sequence('hounde'))
    print(model.correct_sequence('gril'))
    print(model.correct_sequence('knowledegble'))
    print(model.correct_sequence('imediately'))
    print(model.correct_sequence('touhgt'))

    with open("words_alpha.txt") as word_file:
        dictionary = set(word.strip().lower() for word in word_file)
    with open('./datasets/english_misspellings_test.txt') as f:
        test_data = f.read().splitlines()

    model.test_model(test_data, dictionary)
