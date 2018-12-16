# model.py

"""
All-in-one module, with some routines outsourced to helpers.py
"""

from collections import Counter
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import nltk
import os
from os.path import isfile
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

from helpers import clean_text, tsne_plot
from helpers import (get_avg_word_length, get_num_contractions, get_num_adverbs,
                    get_ratio_and_to_comma, get_coordinating_conjunctions)

"""
CLASS OBJECTS
"""
class HemingwaySentences:
    """
    Memory-friendly generator that supports input of a variable number of
    Hemingway files. No proper input check: used internally. USED FOR WORD2VEC.

    - Does not discriminate based on text or period when written: dumps all
    text into the model blindly

    NOTES:
    - Does not encode text as lowercase
    - Splits on sentences, not EOL characters
    """
    def __init__(self, *fnames):
        if len(fnames) == 0:
            self.fnames = ['total.txt']
        else:
            self.fnames = fnames

    def __iter__(self):
        for fname in self.fnames:
            with open('text/hemingway/{}'.format(fname), 'rb') as f:
                # use 'rb' to avoid decode attempt on read, which would stop
                # at unrecognizable characters
                raw = f.read().decode('utf-8', 'ignore') # immediately decode to string
                word_count = 0
                for sentence in raw.split('.'): # split on periods between sentences
                    word_count += len(sentence.split())
                    words = nltk.word_tokenize(clean_text(sentence))
                    yield words # yield sentence as list of words
                print('TRAINING ON A CORPUS OF {} WORDS'.format(word_count))

"""
ROUTINES
"""
def get_model(author):
    if not isfile('models/{}.model'.format(author)):
        hem_sentences = HemingwaySentences()
        """
        NOTES ON Word2Vec PARAMETERS:
            - Using CBOW, the default 'sg'
            - Using 4 workers to reflect 4 cores on 2.5 Ghz i7 Processor
            - Using 'size=200', the default dimensionality of each word vector
            - Using 'window=5', the default distance between target word and surrounding words
            - Using 'min_count=200' to ensure the model is not being trained on outliers.
        """
        model = Word2Vec(hem_sentences, workers=4, min_count=200)
        return model

    return Word2Vec.load('models/{}.model'.format(author))

def create_dataframe(author):
    # ROUTINE TO READ CORPUS INTO PANDAS DATAFRAME

    train = pd.read_csv('text/{}/total.csv'.format(author))
    # TODO: INSERT TESTING CORPUS HERE

    # FEATURES
    # 1: SENTENCE LENGTH
    train['sentence_length'] = [len(str(x).split()) for x in train['sentence']]
    # 2: AVERAGE WORD LENGTH
    train['avg_word_length'] = [get_avg_word_length(str(x)) for x in train['sentence']]
    # 3: NUMBER OF CONTRACTIONS
    train['num_contractions'] = [get_num_contractions(str(x)) for x in train['sentence']]
    # 4: NUMBER OF ADVERBS
    train['num_adverbs'] = [get_num_adverbs(str(x)) for x in train['sentence']]
    # 5: RATIO OF AND'S TO COMMAS
    train['ratio_and_to_commas'] = [get_ratio_and_to_comma(str(x)) for x in train['sentence']]
    # 6: NUMBER OF COORDINATING CONJUNCTIONS PER SENTENCE
    train['num_coordinating_conjunctions'] = [get_coordinating_conjunctions(str(x)) for x in train['sentence']]

    return train

def create_wordcloud(dataset, plotname='hemingway'):
    total = ''.join(str(dataset['sentence']))
    wc = WordCloud(width=1000, height=500, stopwords=STOPWORDS).generate(total)

    plt.figure(figsize=(15,5))
    plt.imshow(wc)
    plt.axis('off')
    plt.title('Word Cloud for {}'.format(plotname))
    plt.savefig('metrics/hemingway/wordcloud_{}.png'.format(plotname))

def get_vocabulary(model):
    ctr = Counter()
    for word, vocab_obj in model.wv.vocab.items():
        ctr[word] = vocab_obj.count
    return ctr

"""
MAIN METHOD
"""
if __name__ == '__main__':
    model = get_model('hemingway')
    model.save('models/hemingway.model')
    tsne_plot(model, 'hemingway')

    create_wordcloud(create_dataframe('hemingway'), plotname='Hemingway')

    # METRICS
    # record top 200 vocab words
    author = 'hemingway'
    with open('metrics/{}/vocab_{}.txt'.format(author, author), 'w') as f:
        f.write("----VOCABULARY OF 'word2vec_{}.model'----\n"
                "LENGTH: {} \n"
                "\n\n\n\n".format(author, len(model.wv.vocab)))

        vocab = get_vocabulary(model)
        for word, count in vocab.most_common(200):
            f.write('----{}----\n{}\n\n'.format(word.strip(), count))
