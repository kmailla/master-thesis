import gensim
import smart_open
import numpy as np
from keras.layers import Embedding, LSTM, Dense, Concatenate, GRU, Input, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.summarization.textcleaner import split_sentences
from keras.models import load_model
import time
from keras import optimizers
import tensorflow as tf
from random import shuffle
import os


# this is for the small version
NUM_EPOCHS_LANG_MODEL = 15
EMBEDDING_SIZE = 100

BACKPROP_WINDOW = 12
NUM_EPOCHS_NETWORK = 30
HIDDEN_UNITS = 100


# there should be stripping of stopwords
def read_data_file(file_name, split_point=False):
    """Reads the sentences that will be used for training

    :param file_name: the path to the data the model will be trained on
    :param split_point: a ratio where the given data file should be seperated into training and validation data
    """
    print('Reading: ' + file_name)
    full_sentences = []
    training_sequences = []
    validation_sequences = []
    lengths = []

    with smart_open.open(file_name, encoding="utf8") as f:
        for i, line in enumerate(f):
            all_sentences = split_sentences(line)
            for sentence in all_sentences:
                sentence_all_tokens = gensim.utils.simple_preprocess(sentence, min_len=1)

                full_sentences.append(sentence_all_tokens)
                lengths.append(len(sentence_all_tokens))

    if split_point:
        split_point = int(len(full_sentences)*0.15)
        shuffle(full_sentences)
        validation_data = full_sentences[0:split_point]
        training_data = full_sentences[split_point:]

        for v in validation_data:
            sentence_sequences = sentence_to_sequences(v)
            validation_sequences.extend(sentence_sequences)
    else:
        training_data = full_sentences

    print('Max length of sentences: ' + str(max(lengths)))
    print('Average length of sentences: ' + str(sum(lengths) / len(lengths)))

    for t in training_data:
        sentence_sequences = sentence_to_sequences(t)
        training_sequences.extend(sentence_sequences)

    print('Number of sentences:', len(full_sentences))

    if split_point:
        return full_sentences, training_sequences, validation_sequences
    else:
        return full_sentences, training_sequences


def sentence_to_sequences(sentence):
    """Chunks the long sentences into smaller ones based on the backpropagation window

    :param sentence: the sentence to be chunked
    :returns a list of sequences that are the sentence divided by the window
    """
    sequences = []
    while len(sentence) > BACKPROP_WINDOW:
        sequence = sentence[0:BACKPROP_WINDOW]
        sequences.append(sequence)
        sentence = sentence[BACKPROP_WINDOW - 1:]
    if len(sentence) > 3:
        sequences.append(sentence)

    return sequences


class WordVectorsModel:
    """"The model that creates word embeddings"""

# Static method of the class. When we load pre-trained vectors, we do not need to create a WordVectorsModel object
    @staticmethod
    def load_vectors(path):
        """Loads existing word vectors

        :param path: the path to the word vectors
        :returns the word vectors
        """
        # bin for pretrained only
        if '.bin' in path:
            embeddings = gensim.models.fasttext.load_facebook_vectors(path)
        else:
            embeddings = gensim.models.FastText.load(path)
        word_vectors = embeddings.wv
        print('Word vectors loaded.')
        return word_vectors

    @staticmethod
    def create_network_input(data, vocabulary):
        """Create the input/output tensors for the prediction model

        :param data: the data the model will be trained on
        :param vocabulary: the dictionary from the word vectors
        :returns predictors and labels (x and y) that can be used for training
        """
        # senteces with words --->
        # number sequences --->
        # fill with zeros to fix size --->
        # predictor and label: the sequence until the latest word + the latest word itself
        x = np.zeros([len(data), BACKPROP_WINDOW], dtype=np.int32)
        y = np.zeros([len(data)], dtype=np.int32)
        for i, sentence in enumerate(data):
            for j, word in enumerate(sentence[:-1]):
                try:
                    x[i, j] = vocabulary[word].index  # 'vocabulary' was 'self.embeddings_model.wv.vocab'
                except KeyError:
                    pass
            try:
                y[i] = vocabulary[sentence[-1]].index
            except KeyError:
                pass

        return x, y

    def __init__(self, num_epochs, num_sentences, embedding_size):
        """Initializing the prediction model

        :param num_epochs: number of epochs for training the word vectors
        :param num_sentences: number of sentences to be used when training
        :param embedding_size: the dimensionality of the word vectors
        """
        # min count 1 - so it won't leave out any words
        # (rare words should be projected into unknown/name/etc. at preprocessing instead)
        self.embeddings_model = gensim.models.FastText(size=embedding_size, window=6, min_count=1)

        self.num_epochs = num_epochs
        self.num_sentences = num_sentences

    def save_model(self, name):
        """Saves the word vector model

        :param name: the name the model will be saved as
        """
        self.embeddings_model.save(os.path.join('saved_models', 'word_prediction', name))

    def train_word_vectors(self, data):
        """Train the word vectors

        :param data: the data the model will be trained on
        :returns the ready made word vectors
        """
        self.embeddings_model.build_vocab(sentences=data)
        self.embeddings_model.train(sentences=data, total_examples=self.num_sentences, epochs=self.num_epochs)

        word_vectors = self.embeddings_model.wv

        return word_vectors


class WordPredictor:
    """"The model that is used for word_prediction.
    The class has variables so it is easier to compare the models with different settings."""

    @staticmethod
    def load_model(model_name):
        """Loads an existing prediction model

        :param model_name: the name of the saved model
        :returns the loaded model
        """
        # load weights into new model
        loaded_model = load_model(model_name + '.h5')
        model = loaded_model
        print('Word prediction model loaded.')

        return model

    def __init__(self, model_type, num_epochs, batch_size, embedding_size, attention=False):
        """Initializing the prediction model

        :param model_type: type of the model, gru/lstm
        :param num_epochs: the path to the sentence pairs
        :param batch_size: number of samples to train with at once
        :param embedding_size: dimensionality of the word vectors
        :param attention: flag for using the attention mechanism
        """
        self.model_type = model_type

        if model_type == 'lstm':
            self.network = LSTM(units=HIDDEN_UNITS, dropout=0.1, return_state=True)
        else:
            self.network = GRU(units=HIDDEN_UNITS, dropout=0.1, return_state=True)

        self.num_epochs = num_epochs
        self.attention = attention
        self.batch_size = batch_size
        self.embedding_size = embedding_size

    def save_model(self, model_name):
        """Saves the prediction model

         :param model_name: name of the model that will be used in the filename
         """
        self.model.save('./saved_models/word_prediction/' + model_name + '.h5')
        print('Saved model to disk')

    def train_network(self, vocab_size, embed_weights, predictors, labels, valid_x=None, valid_y=None,
                      test_x=None, test_y=None, attention=False, opt=None):
        """Training the prediction model

        :param vocab_size: size of the vocabulary of words
        :param embed_weights: the word vectors
        :param predictors: the previous words
        :param labels: the target words
        :param valid_x: validation data's previous words
        :param valid_y: validation data's target words
        :param test_x: test data's previous words
        :param test_y: test data's target words
        :param attention: flag for using the attention mechanism
        :param opt: the name of the optimizer used for training
        """
        t = time.time()
        inputs = Input(shape=(BACKPROP_WINDOW,))
        # embedding layer so we use the word embeddings, not only their indexes from the dictionary
        embed_weights_matrix = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE,
                                         weights=[embed_weights], trainable=False)(inputs)
        attn_name = ''

        if self.model_type == 'lstm':
            logits, states, _ = self.network(embed_weights_matrix)
        else:
            logits, states = self.network(embed_weights_matrix)

        if attention:
            attn_name = '_attention'

            context_vector = Lambda(self.calculate_attention_score)([logits, states])
            logits = Concatenate(axis=-1)([context_vector, logits])

            attention_dense = Dense(EMBEDDING_SIZE, activation='tanh')
            logits = attention_dense(logits)

        softmax_out_layer = Dense(units=vocab_size, activation='softmax')
        softmax_out_probabilities = softmax_out_layer(logits)

        self.model = Model(inputs, softmax_out_probabilities)
        
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(filepath=str("./saved_models/" + self.model_type + attn_name + "_best_model.h5"),
                                     verbose=0, save_best_only=True, monitor='val_loss')
        if valid_x is not None:
            self.model.fit(predictors, labels, verbose=2, batch_size=self.batch_size,
                           epochs=self.num_epochs, validation_data=(valid_x, valid_y),
                           callbacks=[early_stop, checkpoint])
        else:
            self.model.fit(predictors, labels, verbose=2, batch_size=self.batch_size,
                           epochs=self.num_epochs, validation_split=0.1, callbacks=[early_stop])
        print(self.model.summary())

        if test_x is not None:
            scores = self.model.evaluate(test_x, test_y, batch_size=self.batch_size)
            print('Scores on test dataset: ' + str(scores))

        elapsed_time = time.time() - t
        print('Time spent on training predictor network: ', elapsed_time)
        print(' ')

    def calculate_attention_score(self, x):
        """Equations for the attention mechanism

        :param x: outputs and hidden state wrapped in one variable
        :returns the context vector
        """
        outputs, hidden_state = x
        W = Dense(self.embedding_size)
        hidden_with_time_axis = tf.expand_dims(hidden_state, 1)
        score = Dense(1)(tf.nn.tanh(W(outputs) + W(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def test_model(self, test_x, test_y):
        """Test the prediction model on loss and accuracy

        :param test_x: previous words
        :param test_y: target words
        """
        scores = self.model.evaluate(test_x, test_y, batch_size=self.batch_size)
        print(self.model.metrics_names)
        print('Scores on test dataset: ' + str(scores))


def execute(path_saved_wordvectors='./saved_models/word_prediction/wiki.en.bin'):
    corpus_file_name = './datasets/wikitext-2/wiki.train.tokens'
    full_sentences, train_sentences = read_data_file(corpus_file_name)
    full_valid_sentences, valid_sentences = read_data_file('./datasets/wikitext-2/wiki.valid.tokens')
    full_test_sentences, test_sentences = read_data_file('./datasets/wikitext-2/wiki.test.tokens')

    all_sentences = full_sentences + full_valid_sentences + full_test_sentences

    if path_saved_wordvectors:
        word_vectors = WordVectorsModel.load_vectors(path_saved_wordvectors)
        flat_list = [item for sublist in all_sentences for item in sublist]
        used_vocab = list(set(flat_list))
        weights = []
        vocabulary = {}
        i = 0
        # instead of using 3 million word vectors in the embedding layer,
        # pick out the ones that are actually present in the text
        for v in used_vocab:
            if v in word_vectors.vocab.keys():
                v_index = word_vectors.vocab[v].index
                weights.append(word_vectors.syn0[v_index])
                vocabulary[v] = word_vectors.vocab[v]
                vocabulary[v].index = i
                i = i + 1
        weights = np.asarray(weights)
        vocabulary = word_vectors.vocab
        vocab_size, _ = weights.shape
    else:
        embed = WordVectorsModel(num_epochs=NUM_EPOCHS_LANG_MODEL,
                                 num_sentences=len(all_sentences), embedding_size=EMBEDDING_SIZE)
        word_vectors = embed.train_word_vectors(all_sentences)
        weights = word_vectors.syn0
        vocabulary = word_vectors.vocab
        vocab_size, _ = weights.shape

    print("Number of word vectors i.e. words in the vocabulary: " + str(len(vocabulary)))

    train_x, train_y = WordVectorsModel.create_network_input(train_sentences, vocabulary)
    valid_x, valid_y = WordVectorsModel.create_network_input(valid_sentences, vocabulary)
    test_x, test_y = WordVectorsModel.create_network_input(test_sentences, vocabulary)
    opt = optimizers.Adagrad()

    predict = WordPredictor(model_type='gru', batch_size=128, num_epochs=NUM_EPOCHS_NETWORK,
                            embedding_size=EMBEDDING_SIZE)
    predict.train_network(vocab_size, weights, train_x, train_y, valid_x, valid_y, test_x, test_y,
                          attention=False, opt=opt)

    predict2 = WordPredictor(model_type='gru', batch_size=128, num_epochs=NUM_EPOCHS_NETWORK,
                             embedding_size=EMBEDDING_SIZE)
    predict2.train_network(vocab_size, weights, train_x, train_y, valid_x, valid_y, test_x, test_y,
                           attention=True, opt=opt)

    predict3 = WordPredictor(model_type='lstm', batch_size=128, num_epochs=NUM_EPOCHS_NETWORK,
                             embedding_size=EMBEDDING_SIZE)
    predict3.train_network(vocab_size, weights, train_x, train_y, valid_x, valid_y, test_x, test_y,
                           attention=False, opt=opt)

    predict4 = WordPredictor(model_type='lstm', batch_size=128, num_epochs=NUM_EPOCHS_NETWORK,
                             embedding_size=EMBEDDING_SIZE)
    predict4.train_network(vocab_size, weights, train_x, train_y, valid_x, valid_y, test_x, test_y,
                           attention=True, opt=opt)


# the script to run to obtain a smaller network that I use in the app
def get_small_gru():
    all_sentences, train_sentences = read_data_file('./datasets/danish_small.txt')
    embed = WordVectorsModel(num_epochs=NUM_EPOCHS_LANG_MODEL, num_sentences=len(all_sentences),
                             embedding_size=EMBEDDING_SIZE)
    word_vectors = embed.train_word_vectors(all_sentences)
    embed.save_model('gru_danish_small')
    weights = word_vectors.syn0
    vocabulary = word_vectors.vocab
    vocab_size, _ = weights.shape
    
    train_x, train_y = WordVectorsModel.create_network_input(train_sentences, vocabulary)
    opt = optimizers.Adam()
    predict = WordPredictor(model_type='gru', batch_size=64, num_epochs=NUM_EPOCHS_NETWORK,
                            embedding_size=EMBEDDING_SIZE)
    predict.train_network(vocab_size, weights, train_x, train_y, attention=False, opt=opt)
    predict.save_model('gru_danish_small')


def load(path_vectors, path_predictor):
    word_vectors = WordVectorsModel.load_vectors(path_vectors)
    model = WordPredictor.load_model(path_predictor)
