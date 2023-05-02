#!/usr/bin/env python

import sqlite3
import pandas
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, TextVectorization

db = sqlite3.connect('enron.sqlite')
emails_df = pandas.read_sql('select * from enron', db)
emails_df['spaminess'] = np.where(
    emails_df.spam_or_ham == 'spam', 1.0, 0.0)

pseudo_train_data, test_data = train_test_split(emails_df)
train_data, validation_data = train_test_split(pseudo_train_data)

vectorizer = TextVectorization(output_mode='tf_idf', ngrams=2)
vectorizer.adapt(train_data.email_text)
train_vectors = vectorizer(train_data.email_text)
validation_vectors = vectorizer(validation_data.email_text)
test_vectors = vectorizer(test_data.email_text)

vocab_size = vectorizer.vocabulary_size()
inputs = keras.Input(shape=(vocab_size,))
output = Dense(1, activation="sigmoid")(inputs)
model = keras.Model(
    inputs=[inputs],
    outputs=[output])

model.compile(
    loss='binary_crossentropy',
    metrics=["accuracy"])

history = model.fit(
    x=train_vectors,
    y=train_data.spaminess,
    validation_data=(validation_vectors, 
                     validation_data.spaminess),
    callbacks=[keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3
    )],
    verbose=0,
    epochs=20)

vocab = vectorizer.get_vocabulary()
weights = model.get_weights()[0][:,0]

# Put them into a Pandas series so we can manipulate
# them easily.
strengths = pd.Series(index=vocab, data=weights)

print(strengths.nlargest(5))
print(strengths.nsmallest(5))
