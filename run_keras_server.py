import numpy as np
import flask
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from character_seq2seq import Seq2seqModel as CharSeq2seqModel
from word_seq2seq import Seq2seqModel as WordSeq2seqModel
import gensim
from googletrans import Translator
import json
import tensorflow as tf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

eng_word_pred_model = None
eng_word_vectors = None
eng_misspelling_model = None
eng2dan_translation_model = None
eng_prediction_list = {}

dan_word_pred_model = None
dan_word_vectors = None
dan_misspelling_model = None
dan2eng_translation_model = None
dan_prediction_list = {}

BACKPROP_WINDOW = 12
danish_dict = []
translator = Translator()

PREVIOUS_WORDS = None

BASE_RESPONSE = {}

# flag for using second language models as well
use_second_language = True


def load_misspelling_model(data_path, model_path, lang='eng'):
    """Loads the spell correction model

    :param data_path: the path to the data the model was trained on
    :param model_path: path to the model
    :param lang: the language the model was trained on
    """
    global eng_misspelling_model, dan_misspelling_model
    if lang == 'eng':
        eng_misspelling_model = CharSeq2seqModel()
        eng_misspelling_model.load_all(data_path, model_path, 'gru')
    else:
        dan_misspelling_model = CharSeq2seqModel()
        dan_misspelling_model.load_all(data_path, model_path, 'gru')
    print(lang + ' misspelling model loaded.')


def load_prediction_model(path, lang='eng'):
    """Loads the word prediction model

    :param path: the path to the model
    :param lang: the language the model was trained on
    """
    # load the pre-trained Keras model
    global eng_word_pred_model, dan_word_pred_model
    if lang == 'eng':
        eng_word_pred_model = load_model(path)
    else:
        dan_word_pred_model = load_model(path)
    print(lang + ' word prediction model loaded.')


def load_word_vectors(path, lang='eng'):
    """Loads the word vectors for the word prediction model

    :param path: the path to the model
    :param lang: the language the model was trained on
    """
    global eng_word_vectors, dan_word_vectors
    if '.bin' in path:
        embeddings = gensim.models.fasttext.load_facebook_vectors(path)
    else:
        embeddings = gensim.models.FastText.load(path)
    if lang == 'eng':
        eng_word_vectors = embeddings.wv
    else:
        dan_word_vectors = embeddings.wv
    print(lang + ' word vectors loaded.')


def load_translation_model(model_path, data_path, lang='eng'):
    """Loads the translation model

    :param data_path: the path to the data
    :param model_path: path to the model
    :param lang: the language the model translates FROM
    """
    global danish_dict, eng2dan_translation_model, dan2eng_translation_model
    if lang == 'eng':
        eng2dan_translation_model = WordSeq2seqModel()
        eng2dan_translation_model.load_all(model_path, data_path, 'lstm', 'rmsprop', True)
    else:
        dan2eng_translation_model = WordSeq2seqModel()
        dan2eng_translation_model.load_all(model_path, data_path, 'lstm', 'rmsprop', False)


def prepare_words(text, lang='eng'):
    """Converts text to word indices

    :param text: the sentence to be converted
    :param lang: the language of the word vectors
    :returns the word indices based on the word vector model's dictionary
    """
    text = str(text)
    if lang == 'eng':
        indices = [eng_word_vectors.wv.vocab[word].index for word in text.lower().split()]
    else:
        indices = [dan_word_vectors.wv.vocab[word].index for word in text.lower().split()]
    # return the processed word
    return indices


@app.route("/predict", methods=["POST"])
def generate_recommendations():
    """The endpoint that can be called with the previous words and current word.

    :returns the word recommendations and extra variables for debugging/analysis to the user"""
    json_format = flask.request.get_json(force=True)
    json_format = json.loads(json_format)

    previous_words = json_format["previous_words"]
    current_word = json_format["current_word"]

    BASE_RESPONSE["success"] = False

    # if the previous words exist or they have changed since last time
    if previous_words != PREVIOUS_WORDS and len(previous_words) > 0:
        predict(previous_words, BASE_RESPONSE)
    # if there is a current word and previous words were present as well
    if len(eng_prediction_list) > 0 and len(current_word) > 0:
        filter_list(current_word, BASE_RESPONSE)
    return flask.jsonify(BASE_RESPONSE)


def predict(previous_words, response):
    """Creates word predictions based on the previous words, on two languages

    :param previous_words: the previous words typed by the user
    :param response: a json variable with the predictions and extra variables
    """
    response["success"] = False
    global graph, eng_prediction_list, dan_prediction_list, use_second_language, PREVIOUS_WORDS
    with graph.as_default():
        PREVIOUS_WORDS = previous_words
        # preprocess the text to get word vector indices
        indexes = prepare_words(previous_words)
        # use the model and then initialize the list of predictions
        word_list = pad_sequences([indexes], maxlen=BACKPROP_WINDOW, padding='pre')
        probabilities = eng_word_pred_model.predict(x=np.array(word_list))[0]
        eng_data = []
        response["predictions"] = []

        length = len(probabilities)

        for i in range(0, length):
            r = {"word": eng_word_vectors.wv.index2word[i], "probability": float("{0:.10f}".format(probabilities[i]))}
            eng_data.append(r)

        sorted_list = sorted(eng_data, key=lambda k: k.get('probability', 0), reverse=True)
        response["predictions"].extend(sorted_list)

        eng_prediction_list = response["predictions"]
        response["english_predictions"] = response["predictions"][0:10]

        if use_second_language:
            # translation
            translated_text = eng2dan_translation_model.decode_sequence(previous_words)
            translated_indexes = prepare_words(translated_text, lang='dan')
            translated_word_list = pad_sequences([translated_indexes], maxlen=BACKPROP_WINDOW, padding='pre')
            translated_probabilities = dan_word_pred_model.predict(x=np.array(translated_word_list))[0]
            dan_data = []

            length = len(translated_probabilities)

            for i in range(0, length):
                r = {"word": dan_word_vectors.wv.index2word[i],
                     "probability": float("{0:.10f}".format(translated_probabilities[i]))}
                dan_data.append(r)

            dan_prediction_list = sorted(dan_data, key=lambda k: k.get('probability', 0), reverse=True)

            response["danish_predictions"] = dan_prediction_list[0:10]
            response["danish_translation"] = translated_text

        # indicate that the request was a success
        response["success"] = True


def filter_list(current_word, response):
    """Loads the spell correction model

    :param current_word: the word that is currently being written by the user
    :param response: a json variable with the predictions and extra variables
    """
    response["success"] = False

    global graph
    global eng_prediction_list, dan_prediction_list
    with graph.as_default():
        # filter based on the partial word the user typed
        filtered_list = [x for x in eng_prediction_list if x['word'].startswith(current_word)]

        # so it only returns one possible correction...
        # the norvig algorithm can be used if we need a list
        correction = eng_misspelling_model.correct_sequence(current_word)

        matches = [x for x in eng_prediction_list if x['word'].startswith(correction)]
        if len(matches) > 0:
            for match in matches:
                if match not in filtered_list:
                    filtered_list.append(match)

        if use_second_language:
            # translation part
            dan_filtered_list = [x for x in dan_prediction_list if x['word'].startswith(current_word)]
            dan_correction = dan_misspelling_model.correct_sequence(current_word)
            dan_matches = [x for x in dan_prediction_list if x['word'].startswith(dan_correction)]
            if dan_matches:
                for match in dan_matches:
                    if match not in dan_filtered_list:
                        dan_filtered_list.append(match)
            top_suggestions = sorted(dan_filtered_list, key=lambda k: k.get('probability', 0), reverse=True)[0:10]
            # so now get the top offers translated
            for offer in top_suggestions:
                # here it could be my seq2seq as well
                eng_word = translator.translate(offer["word"], src='da', dest='en').text.lower()
                originals = [x for x in filtered_list if x["word"] == eng_word]
                if originals:
                    for o in originals:
                        # overwriting if the danish probability is higher
                        if offer["probability"] > o["probability"]:
                            o["probability"] = offer["probability"]
                else:
                    filtered_list.append({"word": eng_word, "probability": offer["probability"]})
            response["danish_correction"] = dan_correction
            response["danish_match"] = dan_matches

        # response["DEBUG"] = filtered_list
        response["predictions"] = sorted(filtered_list, key=lambda k: k.get('probability', 0), reverse=True)

        response["predictions"] = response["predictions"][0:10]
        response["match"] = matches
        response["correction"] = correction
        response["success"] = True


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_word_vectors('./saved_models/word_prediction/gru_english_small', lang='eng')
    load_word_vectors('./saved_models/word_prediction/gru_danish_small', lang='dan')

    load_prediction_model('./saved_models/word_prediction/gru_english_small.h5', lang='eng')
    load_prediction_model('./saved_models/word_prediction/gru_danish_small.h5', lang='dan')

    load_misspelling_model('misspelling_gru_512_eng', './datasets/english_misspellings_train.txt', lang='eng')
    load_misspelling_model('misspelling_gru_512_dan', './datasets/danish_misspellings_train.txt', lang='dan')

    load_translation_model('lstm_64_256_english_danish', './datasets/danish-english_train.txt',
                           './datasets/danish_words.txt')

    graph = tf.get_default_graph()
    app.config['JSON_AS_ASCII'] = False
    app.run(port=5002)
