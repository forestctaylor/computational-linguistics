# helpers.py

"""
Module containing routines used elsewhere in computational-linguistics.
"""

from __future__ import absolute_import

import matplotlib.pyplot as plt
import re
from sklearn.manifold import TSNE
import statistics

"""
ROUTINES
"""
def clean_text(text):
    """
    Helper deciding what to eliminate in a text corpus.
    """
    return text

def get_avg_word_length(str):
    words = str.split()
    if len(words) == 0: # EDGE CASE
        return 0

    sum = 0
    for word in words:
        sum += len(word)
    length = len(words)
    return sum/float(length)

def get_num_adverbs(str):
    # COUNT THE NUMBER OF WORDS THAT END IN '-ly'
    num_adverbs = 0
    for word in str.split():
        if word[-2:] == 'ly':
            num_adverbs += 1
    return num_adverbs

def get_num_contractions(str):
    # RETURN THE NUMBER OF CONTRACTIONS IN THE SENTENCE
    # NOTE: ASSUMES APOSTROPHES HAVE BEEN REMOVED
    contractions = ['aint', 'arent', 'cant', 'couldve', 'couldnt', 'couldntve',
                    'darent', 'daresnt', 'dasnt', 'didnt', 'doesnt', 'dont',
                    'everyones', 'gimme', 'gonna', 'gotta', 'hadnt', 'hasnt',
                    'havent', 'hed', 'hes', 'heve', 'howd', 'howll', 'howre',
                    'hows', 'Id', 'Ill', 'Im', 'Ive', 'itll', 'its', 'lets',
                    'mightve', 'mustve', 'oer', 'oughtnt', 'shant', 'shed',
                    'shell', 'shes', 'shouldve', 'shouldnt', 'thatll', 'thatre',
                    'thats', 'thatd', 'thered', 'therell', 'therere', 'theres',
                    'theyd', 'theyll', 'theyre', 'theyve', 'thiss', 'thosre',
                    'tis', 'twas', 'wasnt', 'wed', 'well', 'were', 'weve',
                    'werent', 'whatd', 'whatll', 'whatre', 'whod', 'wholl',
                    'whyd', 'whyre', 'whys', 'wont', 'wouldve', 'wouldnt',
                    'yall', 'youd', 'youll', 'youre', 'youve', 'heres']

    num_contractions = 0
    for word in str.split():
        if word in contractions:
            num_contractions += 1
    return num_contractions

def get_coordinating_conjunctions(str):
    conj = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so']
    num_conj = 0
    for word in str.split():
        word = re.sub(r'[^a-zA-Z]', '', word.strip())
        if word in conj:
            num_conj += 1
    return num_conj

def get_ratio_and_to_comma(str):
    num_ands = 1
    num_commas = 1
    for word in str.split():
        if word == 'and':
            num_ands += 1
        if ',' in word:
            num_commas += 1
    return num_ands/float(num_commas)

def compare(model, data):
    """
    Algorithm to measure how likely a paragraph is to be written by a particular
    author.
    """
    # METRIC USING FEATURES
    sentences = data.split('.')
    noperiods = data.replace('.', ' ')
    words = noperiods.split()

    sentence_lengths = [len(x.split()) for x in sentences]
    word_lengths = [len(x) for x in words]
    num_contractions = [get_num_contractions(x) for x in sentences]
    num_adverbs = [get_num_adverbs(x) for x in sentences]
    ratio_and_to_commas = [get_ratio_and_to_comma(x) for x in sentences]
    num_coordinating_conjunctions = [get_coordinating_conjunctions(x) for x in sentences]

    df_confidence = 1

    # METRIC USING WORD2VEC, COSINE SIMILARITY
    num_similar_words, sim = 1, 0
    for sentence in sentences:
        words = sentence.split()
        for idx, word in enumerate(words):
            # Check if sentence contains one of top-10 most similar words for each word in sentence
            try:
                similar_words = model.wv.similar_by_word(word, topn=max(sentence_lengths))
            except KeyError:
                continue
            for similar_word in similar_words:
                if similar_word[0] in sentence:
                    num_similar_words += 1

            # Check cosine similarity to immediately neighboring words
            word_back, word_next = None, None
            if idx - 1 > 0:
                word_back = words[idx-1]
                try:
                    word_back_sim = model.wv.similarity(word_back, word)
                except KeyError:
                    continue
            if idx + 1 < len(words) - 1:
                word_next = words[idx + 1]
                try:
                    word_next_sim = model.wv.similarity(word_next, word)
                except KeyError:
                    continue
            if word_back and word_next:
                sim = statistics.mean([float(word_back_sim), float(word_next_sim)])
            elif word_back and not word_next:
                sim = word_back_sim
            elif word_next and not word_back:
                sim = word_next_sim
            # else case handled by default sim value of 0

    m = [(1 - 1/float(num_similar_words)), (float(sim)/(1 - sim))]
    wv_confidence = statistics.mean(m)

    #return statistics.mean([df_confidence, wv_confidence])
    return wv_confidence

def tsne_plot(model, author):
    """
    Creates and saves a scatterplot for the model to 'model.png'
    """
    all_labels = [word for word in model.wv.vocab]
    labels = []
    for word in all_labels:
        if model.wv.vocab[word].count > 50:
            labels.append(word)
    tokens = [model[word] for word in labels]

    tsne_model = TSNE()
    new_values = tsne_model.fit_transform(tokens)

    x = [value[0] for value in new_values]
    y = [value[1] for value in new_values]

    plt.figure(figsize=(16,16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i], xy=(x[i], y[i]))

    plt.savefig('metrics/{}/model_{}.png'.format(author, author))
