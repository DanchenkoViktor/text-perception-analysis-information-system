import gensim
import pandas as pd
import numpy as np
import tensorflow as tf

from FileUtils import csv_dataset, Author
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

#print(csv_dataset())

train = pd.read_csv('information system/dataset/dataset.csv', sep=',')

print(train.head(15))
print(train.groupby('key').nunique())

train = train[['sentence','key']]
#print(train["sentence"].isnull().sum())

labels = np.array(train['key'])
y = []
for i in range(len(labels)):
    if labels[i] == Author.Orwell_George.value:
        y.append(0)
    if labels[i] == Author.Tolstoy_Lev_Nikolayevich.value:
        y.append(1)
    if labels[i] == Author.Zamyatin_Evgeny_Ivanovich.value:
        y.append(2)
    if labels[i] == Author.Pushkin_Alexander_Sergeevich.value:
        y.append(3)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 4, dtype="int")
del y

#print(labels)

max_len = 200

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


sequences = np.array(train['sentence'])
print(list(sent_to_words(sequences))[:20])



#tweets = pad_sequences(sequences, maxlen=max_len)
#print(tweets)
#print(dict.get('Zamyatin_Evgeny_Ivanovich'))
#print(dict_all_dataset.get('Zamyatin_Evgeny_Ivanovich'))
#get_statistic_words_from_text(get_tokens_from_text(dict_all_dataset.get('Zamyatin_Evgeny_Ivanovich')))
