import pickle

from keras import layers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Sequential

from FileUtils import csv_dataset, Author, remove_dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

print('Done')

# Create dataset as csv file
# csv_dataset()

train = pd.read_csv('dataset/dataset.csv', sep=',')

print(train['key'].unique())
print(train.groupby('key').nunique())

train = train[['sentence', 'key']]
train["sentence"].isnull().sum()
train["sentence"].fillna("No content", inplace=True)

data = np.array(train['sentence'])
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
labels = tf.keras.utils.to_categorical(y, 4, dtype="float32")
del y

print(data)
print(labels)

max_len = 200
max_words = 5000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(tweets, labels, random_state=0)
print(len(X_train), len(X_test), len(y_train), len(y_test))

model1 = Sequential()
model1.add(layers.Embedding(max_words, 20))
model1.add(layers.LSTM(15, dropout=0.5))
model1.add(layers.Dense(4, activation='softmax'))

model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto',
                              save_freq='epoch', save_weights_only=False)
history = model1.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[checkpoint1])

model2 = Sequential()
model2.add(layers.Embedding(max_words, 40, input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(20, dropout=0.6)))
model2.add(layers.Dense(4, activation='softmax'))
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint2 = ModelCheckpoint("best_model2.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto',
                              save_freq='epoch', save_weights_only=False)
history = model2.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[checkpoint2])

from keras import regularizers

model3 = Sequential()
model3.add(layers.Embedding(max_words, 40, input_length=max_len))
model3.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),
                         bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.MaxPooling1D(5))
model3.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),
                         bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.GlobalMaxPooling1D())
model3.add(layers.Dense(4, activation='softmax'))
model3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
checkpoint3 = ModelCheckpoint("best_model3.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto',
                              save_freq='epoch', save_weights_only=False)
history = model3.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[checkpoint3])

best_model = keras.models.load_model("best_model2.hdf5")

test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ', test_acc)

predictions = best_model.predict(X_test)

matrix = confusion_matrix(y_test.argmax(axis=1), np.around(predictions, decimals=0).argmax(axis=1))
sentiment = [
    Author.Orwell_George.name,
    Author.Tolstoy_Lev_Nikolayevich.name,
    Author.Zamyatin_Evgeny_Ivanovich.name,
    Author.Pushkin_Alexander_Sergeevich.name
]

conf_matrix = pd.DataFrame(
    matrix,
    index=sentiment,
    columns=sentiment
)
# Normalizing
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(15, 15))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})

# TEST
sequence = tokenizer.texts_to_sequences(['this experience has been the worst , want my money back'])
test = pad_sequences(sequence, maxlen=max_len)
sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]

# Saving weights and tokenizer so we can reduce training time on SageMaker

# serialize model to JSON
model_json = best_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
best_model.save_weights("model-weights.h5")
print("Model saved")

# saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Tokenizer saved')
