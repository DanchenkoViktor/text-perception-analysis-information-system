import csv
import enum
import os
from os.path import join

from docx2python import docx2python
from os import walk

import Precondition
from Precondition import dataset_preprocessing, save_dataset_to_csv, split_text_on_vectors

PATH_DATASET = 'dataset/'
NAME_DATASET_CSV_FILE = 'dataset.csv'
BATCH_SIZE = 1024
SEED = 123


def get_all_dataset():
    print("DATASET")
    global authors, writings
    try:
        _, authors, _ = next(walk(PATH_DATASET))
    except Warning:
        print("Dataset is empty")
    d = dict()
    d.fromkeys(authors)

    for author in authors:
        path_to_works_of_the_author = join(PATH_DATASET, author)
        try:
            _, _, writings = next(walk(path_to_works_of_the_author))
        except Warning:
            print(author, "'s dataset is empty")
        print("\n")
        print(reformat_names(reformat_name(author)), "writing: ", sep=" ")
        [print(reformat_names(writing), sep='\n') for writing in writings]
        global_vacab = ''
        for filename in writings:
            path = join(path_to_works_of_the_author, filename)
            # extract docx content
            doc_result = docx2python(path).text
            text = dataset_preprocessing(doc_result)
            d.update({author: global_vacab.join(text)})
    return d


def get_df_from_dataset():
    print("DATASET")
    global authors, writings
    try:
        _, authors, _ = next(walk(PATH_DATASET))
    except Warning:
        print("Dataset is empty")
    d = dict()
    d.fromkeys(authors)

    for author in authors:
        path_to_works_of_the_author = join(PATH_DATASET, author)
        try:
            _, _, writings = next(walk(path_to_works_of_the_author))
        except Warning:
            print(author, "'s dataset is empty")
        print("\n")
        print(reformat_names(reformat_name(author)), "writing: ", sep=" ")
        [print(reformat_names(writing), sep='\n') for writing in writings]
        global_vacab = set()
        for filename in writings:
            path = join(path_to_works_of_the_author, filename)
            # extract docx content
            doc_result = docx2python(path).text
            text = dataset_preprocessing(doc_result)
            d.update({author: global_vacab.union(text)})
    return d


def reformat_names(name):
    return name \
        .replace("_", " ") \
        .replace("-", " ") \
        .replace(".docx", "")


def reformat_name(author):
    author_spit = reformat_names(author).split(" ", 1)
    return "{0}'s {1}".format(author_spit[0], author_spit[1])


def csv_dataset():
    remove_dataset()
    create_dataset_file(Precondition.PATH_DATASET_CSV)
    result = f"\nDataset is saved as CSV file"
    dict = get_all_dataset()
    for author in Author:
        try:
            vectors = split_text_on_vectors(dict.get(author.name), 2)
            save_dataset_to_csv(vectors, author.value)
        except Exception:
            return f"\nFail saving dataset on csv"
    return print(result)


def create_dataset_file(path):
    with open(path, mode='w') as csv_file:
        fieldnames = ['textId', 'sentence', 'key']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    return 0


def remove_dataset():
    if os.path.exists('dataset/dataset.csv'):
        os.remove("dataset/dataset.csv")
        return print(f'\nOld dataset is removed\n')
    else:
        return print('Dataset is not exists')


class Author(enum.Enum):
    Orwell_George = 0
    Tolstoy_Lev_Nikolayevich = 1
    Zamyatin_Evgeny_Ivanovich = 2
    Pushkin_Alexander_Sergeevich = 3
