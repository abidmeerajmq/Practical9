#!/usr/bin/env python
import sqlite3
import pandas
db = sqlite3.connect('enron.sqlite')
emails_df = pandas.read_sql('select * from enron', db)
print(f"{emails_df.shape=}")

from tensorflow import keras
vectorizer = keras.layers.TextVectorization()
vectorizer.adapt(emails_df.email_text)
text_vectors = vectorizer(emails_df.email_text)
print(f"{text_vectors.shape=}")
print(text_vectors[:3])
print(vectorizer.get_vocabulary()[:5])
