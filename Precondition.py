import re
import string
import csv


def dataset_preprocessing(dataset):
    spec_chars = string.punctuation + '\’\xa0«»\t-—…‘„“–'
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


def save_dataset_to_csv(vectors, key):
    try:
        with open('information system/dataset/dataset.csv', mode='a') as csv_file:
            fieldnames = ['sentence', 'key']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for vector in vectors:
                writer.writerow({fieldnames[0]: vector, fieldnames[1]: key})
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
    return vectors
