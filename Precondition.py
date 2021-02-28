import re


def clear_text_dataset(dataset):
    ds = custom_standardization(dataset)
    vocabulary = clean_words(ds)
    # Проверка очистки данных
    reg = re.compile('[^A-Za-zА-Я0-9!"№;%:-?*()\']')
    checkText = [word for word in vocabulary if reg.sub('', word)]
    if not checkText:
        print('Успешно! Текст очищен для обучения')
    return vocabulary


# We create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
    kill_punctuation = str.maketrans('', '', r"-—()\"#/@;:<>{}=~|.?!,[]«»")
    output_text = input_data.translate(kill_punctuation).split()
    return output_text


def clean_words(words):
    reg = re.compile('[^а-я]')
    # Drop numbers
    words = [word for word in words if not word.isdigit()]
    # Drop one char word
    words = [word for word in words if len(word) > 1]
    words = [word for word in words if reg.sub('', word)]
    # Drop stopwords
    words = [word for word in words if word[0].isalpha()]
    words = [word.lower() for word in words]
    words = set(words)
    return words
