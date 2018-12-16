# text_statistics.py

"""
Module to analyze statistics a particular file or folder of files.
"""

import hashlib
import os
import pandas as pd

def compile_by_author(author):
    # COMPILES ALL .txt FILES UNDER FILE 'author'

    with open('{}/total.txt'.format(author), 'wb') as total:
        filenames = os.listdir(author) # get list of texts in directory
        for file in filenames:
            with open('{}/{}'.format(author, file), 'rb') as f:
                raw = f.read().decode('utf-8', 'ignore')
                total.write(raw.encode('utf-8'))

    print('WROTE {} FILES TO {}/total.txt'.format(author, author))

def get_number_words(filename):
    # GET THE NUMBER OF WORDS IN A FILE
    with open(filename, 'rb') as f:
        raw = f.read().decode('utf-8', 'ignore')
        words = raw.split()
        print('{} WORDS IN {}'.format(len(words), filename))

def write_to_csv(filename):
    # ROUTINE TO WRITE EACH SENTENCE TO A NEW ROW IN A CSV
    root = filename.split('.')[0]

    with open(filename, 'rb') as f:
        raw = f.read().decode('utf-8')
        raw.replace('?', '.')
        raw.replace('!', '.')
        sentences = raw.split('.')
        dict = {'id': [], 'sentence': []}

        for sentence in sentences: # WRITE ID AND SENTENCE TO DICTIONARY
            sentence.strip() # STRIP LEADING AND TRAILING WHITESPACE
            if len(sentence) < 3: # SKIP USELESS SENTENCES
                continue

            hash_ = hashlib.md5(sentence.encode('ascii', 'ignore')).hexdigest()
            dict['id'].append(hash_[:10])# ID IS [:10] MD5 HASH OF SENTENCE
            dict['sentence'].append(sentence)

        df = pd.DataFrame.from_dict(data=dict)

        with open('{}.csv'.format(root), 'wb') as csvfile:
            df.to_csv(csvfile, encoding='utf-8')
