# app.py

"""
Module containing algorithms to analyze incoming paragraphs.
"""
import argparse
import gensim
from gensim.models import Word2Vec

from collections import Counter
from helpers import compare

"""
COMMAND LINE ARGUMENTS
"""
parser = argparse.ArgumentParser(description='Provide path to text file with input paragraph')
parser.add_argument('filename', default=None,
                    help='input filename')
args = parser.parse_args()

"""
ENVIRONMENT VARIABLES & GLOBAL OBJECTS
"""
data = None
if not args.filename:
    print('NO FILENAME SPECIFIED')
else:
    with open(args.filename, 'rb') as f:
        data = f.read().decode('utf8', 'ignore') # immediately decode to string

models = {}
mhemingway = Word2Vec.load('models/hemingway.model')
models.update({'hemingway': mhemingway})

"""
ROUTINES
"""
comparisons = {}
for author, model in models.items():
    metric = compare(model, data)
    comparisons.update({author: metric})

maxauthor = None
max = -1
for author, metric in comparisons.items():
    if metric > max:
        maxauthor = author
        max = metric

print("THE APPLICATION PREDICTS THE TEXT WAS WRITTEN BY {} WITH A CONFIDENCE METRIC {}".format(maxauthor, metric))
