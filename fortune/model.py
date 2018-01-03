'''
Machine learning model and training code.
'''

import numpy as np
from keras.layers import (LSTM, BatchNormalization, CuDNNLSTM, Dense, Dropout,
                          Embedding, Reshape)
from keras.models import Sequential, Model
from keras.models import model_from_yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class WordLanguageModelVectorizer(BaseEstimator, TransformerMixin):
    '''
    Base language model uses a CharacterEncoder to create character ordinals
    and then applies a transformation in order to create vectors.
    '''

    def __init__(self, context_length=64):
        '''
        Parameters
        ----------
        context_length : int
            This number of words will be used as a context to predict future words.
        '''
        self.context_length = context_length
        self.sequencer = CountVectorizer()

    def fit(self, strings):
        '''
        Fit the word vocabulary to target strings.

        Parameters
        ----------
        strings : iterable
            An iterable of source strings.
        '''
        # forgive passing a single string
        if type(strings) is str:
            strings = [strings]
        self.sequencer.fit(strings)
        self.sequencer.inverse_vocabulary_ = {
            sequence: word for word, sequence in self.sequencer.vocabulary_.items()}
        self.unique_words = len(self.sequencer.inverse_vocabulary_)
        return self

    def transform(self, strings):
        '''
        Transform strings into a (X, Y) pairing.

        Parameters
        ----------
        strings : iterable
            An iterable of source strings.

       Returns
        -------
        (np.ndarray, np.ndarray)
            A tuple (X, Y) three dimensional [sample_index, character_index] context X with a word sequence number
            to be embedded, and a two dimensional [sample_index, one_hot] target Y.
        '''
        # forgive passing a single string
        if type(strings) is str:
            strings = [strings]
        # start off by turning all the text into a series of integers
        word_sequence_numbers = []
        for string in strings:
            as_words = self.sequencer.build_analyzer()(string)
            word_sequence_numbers += list(
                map(self.sequencer.vocabulary_.get, as_words))
        # pad to the minimum context length
        if len(word_sequence_numbers) <= self.context_length:
            word_sequence_numbers = [
                0] * (1 + self.context_length - len(word_sequence_numbers)) + word_sequence_numbers

        # make this number of overlappinq sequences
        # ex with context 2: The quick brown fox likes chickens
        # The quick -> brown
        # quick brown -> fox
        number_of_contexts = len(word_sequence_numbers) - self.context_length
        # sequence numbers for context words
        x = np.zeros((number_of_contexts, self.context_length), dtype=np.int32)
        # one hot encodings for target words
        y = np.zeros((number_of_contexts, self.unique_words), dtype=np.bool)
        for i in range(number_of_contexts):
            context = np.array(
                word_sequence_numbers[i:i + self.context_length])
            x[i] = context
            target = word_sequence_numbers[i + self.context_length]
            y[i, target] = True
        return x, y

    def inverse_transform(self, X):
        '''
        Given a matrix of one hot encodings, reverse the transformation and return a matrix of words.
        '''
        ordinals = X.argmax(-1)
        decoder = np.vectorize(self.sequencer.inverse_vocabulary_.get)
        # allow for single words or lists of words
        decoded = np.array([decoder(ordinals)])
        return ' '.join(decoded.flatten())


class EmbeddedRecurrentLanguageModel(BaseEstimator):
    '''
    Create a language model with a neural network and normalized character encoding.
    '''

    def __init__(self, vectorizer, hidden_layers=256):
        '''
        Parameters
        ----------
        vectorizer : transformer
            Object to transform input strings into numerical encodings.
        hidden_layers : int
            Size of the model's hidden layer, controls complexity.
        '''
        self.vectorizer = vectorizer
        self.hidden_layers = hidden_layers

    def fit(self, strings, epochs=256, batch_size=64):
        '''
        Create and fit a model to the passed in strings.

        Parameters
        ----------
        strings : iterable
            An iterable source of string text.
        '''
        X, Y = self.vectorizer.fit_transform(strings)
        self.model = model = Sequential()
        # begin by embedding character positions
        model.add(Embedding(self.vectorizer.unique_words,
                            self.hidden_layers, input_shape=(X.shape[1],)))
        # and then work on the embeddings recurrently
        model.add(LSTM(self.hidden_layers, return_sequences=True))
        model.add(BatchNormalization())
        model.add(LSTM(self.hidden_layers))
        model.add(BatchNormalization())
        model.add(Dense(self.hidden_layers, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.hidden_layers, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(Y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    def __getstate__(self):
        '''
        Intercept pickling in order to store the model and weights.
        '''
        return {
            'vectorizer': self.vectorizer,
            'hidden_layers': self.hidden_layers,
            'classifier_config': self.model.to_yaml(),
            'classifier_weights': self.model.get_weights()
        }

    def __setstate__(self, state):
        '''
        Restore vectorizer and keras model with weights.
        '''
        self.vectorizer = state['vectorizer']
        self.hidden_layers = state['hidden_layers']
        self.model = model_from_yaml(state['classifier_config'])
        self.model.set_weights(state['classifier_weights'])
