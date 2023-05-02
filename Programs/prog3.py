#!/usr/bin/env python

import sqlite3
import pandas
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
emails_df = pandas.read_sql('select * from enron', db)

emails_df['spaminess'] = np.where(
    emails_df.spam_or_ham == 'spam', 1.0, 0.0)

tv, test_data = train_test_split(emails_df)
train_data, validation_data = train_test_split(tv)

vectorizer = keras.layers.TextVectorization(
    output_mode='tf_idf', ngrams=2)
vectorizer.adapt(train_data.email_text)
train_vectors = vectorizer(train_data.email_text)
validation_vecs = vectorizer(validation_data.email_text)
test_vectors = vectorizer(test_data.email_text)

model = keras.models.Sequential(
    [keras.layers.Dense(1, activation="sigmoid")])

model.compile(
    loss='binary_crossentropy',
    metrics=["accuracy"])

history = model.fit(
    x=train_vectors,
    y=train_data.spaminess,
    validation_data=(validation_vecs, 
                     validation_data.spaminess),
    callbacks=[keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3
    )],
    epochs=20)
print(model.evaluate(test_vectors, test_data.spaminess))
