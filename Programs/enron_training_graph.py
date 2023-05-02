#!/usr/bin/env python

import sqlite3
import pandas
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, TextVectorization
import matplotlib.pyplot as plt

keras.utils.set_random_seed(123456)

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
callbacks = [keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3
    )]

model.compile(loss='binary_crossentropy',
    metrics=["accuracy"])

history = model.fit(
    x=train_vectors,
    y=train_data.spaminess,
    validation_data=(validation_vectors,
                     validation_data.spaminess),
    callbacks=callbacks,
    verbose=0,
    epochs=20)

fig, axes = plt.subplots(ncols=2, figsize=(12,6))
axes[0].plot(history.history['accuracy'],
        label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'],
        label='Validation Accuracy')
axes[0].set_title('Training and Validation Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy (bigger is better)')
axes[0].legend()


axes[1].plot(history.history['loss'],
        label='Training Loss')
axes[1].plot(history.history['val_loss'],
        label='Validation Loss')
axes[1].set_title('Training and Validation Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss (smaller is better)')
axes[1].legend()

fig.savefig("enron_training_graph.png")

print("Corpus size = ", emails_df.shape[0])
print("Vocabulary size =", vocab_size)
print("Corpus : vocabulary ratio =",
      emails_df.shape[0] / vocab_size)
print("Test evaluation:")
print(model.evaluate(test_vectors,
                     test_data.spaminess,
                     return_dict=True,
                     verbose=0))
