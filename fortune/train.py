'''
Train a new machine learning model based on a set of quotes.
'''

from . import model
import pickle
import os

def execute(quote_file_name):
    '''
    Execute the training sequence with the supplied file of quotes.

    Arguments
    ---------
    quote_file_name : str
        Path to the file to read, one quote per line.
    '''

    ### make up a 'special' end of quote word
    with open(quote_file_name, encoding='utf8') as quote_file:
        quotes = list(map(lambda x: x.strip() + ' XXX', quote_file))
        vectorizer = model.WordLanguageModelVectorizer(8)
        learn = model.EmbeddedRecurrentLanguageModel(vectorizer, hidden_layers=32)
        learn.fit(quotes, 1)
        save_to = os.path.join(os.environ['HOME'], '.machine-learning-fortune')
        with open(save_to, mode='wb') as save_file:
            pickle.dump(learn, save_file)