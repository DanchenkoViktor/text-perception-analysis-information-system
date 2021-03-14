from os.path import join

from docx2python import docx2python
from os import walk

from Precondition import clear_text_dataset

PATH_DATASET = 'information system/dataset/'


def get_dict_with_dataset():
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
        print(reformat_names(reformat_name(author)), "writing: ", sep="")
        [print(reformat_names(writing), sep='\n') for writing in writings]
        global_vacab = set()
        for filename in writings:
            path = join(path_to_works_of_the_author, filename)
            # extract docx content
            doc_result = docx2python(path).text
            text = clear_text_dataset(doc_result)
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