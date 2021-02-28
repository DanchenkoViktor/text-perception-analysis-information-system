from os.path import join

from docx2python import docx2python
from os import walk

from Precondition import clear_text_dataset

PATH_DATASET = 'information system/dataset/'
_, authors, _ = next(walk(PATH_DATASET))
print(authors)

d = dict()
d.fromkeys(authors)

for author in authors:
    path_to_works_of_the_author = join(PATH_DATASET, author)
    _, _, filenames = next(walk(path_to_works_of_the_author))

    print(author)
    print(filenames)
    global_vacab = set()
    for filename in filenames:
        path = join(path_to_works_of_the_author, filename)
        # extract docx content
        doc_result = docx2python(path).text
        text = clear_text_dataset(doc_result)
        if len(global_vacab) == 0:
            d.update({author: global_vacab.union(text)})
        else:
            d.update({author: global_vacab.union(text)})
