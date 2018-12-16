# text_process.py

"""
Module with helpers to process, clean text data. Written with file protection.
"""

import re

def remove_page_numbers(filename):
    with open(filename, 'rb') as f: # read access, read as bytes obj
        raw = f.read().decode('utf-8', 'ignore')

    count = len(re.findall(r'[0-9]', raw)) # find number of page numbers
    cleaned = re.sub(r'[0-9]', '', raw) # replace every digit with empty string

    with open(filename, 'w') as f: # write access
        f.write(cleaned)

    print('{} PAGE NUMBERS REMOVED FROM {}'.format(count, filename))

def remove_string(filename, string):
    with open(filename, 'rb') as f: # read access, read as bytes obj
        raw = f.read().decode('utf-8', 'ignore')

    count = raw.count(string)
    cleaned = raw.replace(string, '') # replace every instance with empty string

    with open(filename, 'w') as f: # write access
        f.write(cleaned)

    print('{} INSTANCES OF "{}" REMOVED FROM {}'.format(count, string, filename))

def remove_newlines(filename):
    # REMOVES NEWLINES AND CARRIAGE RETURNS
    with open(filename, 'rb') as f: # read access, read as bytes obj
        raw = f.read().decode('utf-8', 'ignore')

    count = raw.count('\n')
    count += raw.count('\r')
    cleaned = raw.replace('\n', ' ') # replace with space in case words not
    cleaned = cleaned.replace('\r', ' ') # spaced across lines

    with open(filename, 'w') as f: # write access
        f.write(cleaned)

    print('{} NEWLINES REMOVED FROM {}'.format(count, filename))

def remove_non_alphanumerics(filename, str=None):
    """
    Removes all non-alphanumeric characters except:
    ' " , . ? / : ; ! $ & ) ( ) *space* \n
    """
    with open(filename, 'rb') as f: # read access, read as bytes obj
        raw = f.read().decode('utf-8', 'ignore')

    count = len(re.findall(r'[^a-zA-Z0-9\-\.]', raw))
    cleaned = re.sub(r'[^a-zA-Z0-9\-\.]', '', raw)

    with open(filename, 'w') as f: # write access
        f.write(cleaned)

    print('REMOVED {} NON-ALPHANUMERICS FROM {}'.format(count, filename))

def remove_quotations(filename):
    with open(filename, 'rb') as f: # read access, read as bytes obj
        raw = f.read().decode('utf-8', 'ignore')

    # NOTE: ONLY REMOVES ASCII QUOTATIONS, THERE EXIST UNICODE QUOTATIONS
    # THAT ARE NOT ACCOUNTED FOR. USE remove_string() IN THESE CASES
    count = len(re.findall(r'[\'\"\‘\’]', raw))
    cleaned = re.sub(r'[\'\"\‘\’]', '', raw)

    with open(filename, 'w') as f: # write access
        f.write(cleaned)

    print('REMOVED {} QUOTATIONS FROM {}'.format(count, filename))
