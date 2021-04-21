import re
import string
import csv
import uuid

import gensim
from nltk.tokenize.treebank import TreebankWordDetokenizer

PATH_DATASET_CSV = 'dataset/dataset.csv'


def dataset_preprocessing(dataset):
    spec_chars = string.punctuation + '\’\xa0«»\t-—…‘„“–()'
    french_letters_diacritics = 'ÉÂâÊêÎîÔôÛûÀàÈèÙùËëÏïÜüŸÿÇç̀'
    dataset = dataset.lower()
    dataset = dataset.replace('\n', ' ')
    dataset = dataset.replace('о́', 'о')
    dataset = dataset.replace('é', 'е')
    dataset = remove_chars_from_text(dataset, spec_chars)
    dataset = remove_chars_from_text(dataset, string.digits)
    dataset = remove_chars_from_text(dataset, string.ascii_lowercase)
    dataset = remove_chars_from_text(dataset, french_letters_diacritics)
    dataset = re.sub(r'\s+', ' ', dataset)
    dataset = ' '.join(word for word in dataset.split() if len(word) > 3)
    return dataset


def remove_chars_from_text(text, chars):
    return "".join([ch for ch in text if ch not in chars])


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def save_dataset_to_csv(vectors, key):
    try:
        with open(PATH_DATASET_CSV, mode='a') as csv_file:
            fieldnames = ['textId', 'sentence', 'key']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            for vector in vectors:
                writer.writerow({fieldnames[0]: uuid.uuid4().hex[:12], fieldnames[1]: vector, fieldnames[2]: key})
    except Exception:
        print('Dataset doesn\'t save as csv file')


def split_text_on_vectors(text, size_vector):
    array_words = text.split()
    vectors = []
    vector = []
    step = 0
    index = 0
    for element in array_words:
        step = step + 1
        index = index + 1
        vector.append(element)
        if step == size_vector or index == len(array_words):
            if (index - (size_vector + 1)) > 0:
                border_el_left = array_words[index - (len(vector) + 1)]
                vector.insert(0, border_el_left)
            if index < len(array_words):
                border_el_right = array_words[index]
                vector.insert(size_vector + 1, border_el_right)
            vectors.append(" ".join(vector))
            vector = []
            step = 0
    vectors = list(sent_to_words(vectors))
    data = []
    for i in range(len(vectors)):
        data.append(detokenize(vectors[i]))
    print(data[:5])
    return data


def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)
