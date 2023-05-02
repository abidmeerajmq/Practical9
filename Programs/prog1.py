#!/usr/bin/env python

import zipfile
import pandas as pd
import io
import sqlite3

def read_file_from_zip(zf, file_path):
    f = zf.open(file_path, 'r')
    try:
        return io.TextIOWrapper(f, encoding='cp1252').read()
    except UnicodeDecodeError:
        pass
    f = zf.open(file_path, 'r')
    try:
        return io.TextIOWrapper(f, encoding='utf-8').read()
    except UnicodeDecodeError:
        pass
    f = zf.open(file_path, 'r')
    return io.TextIOWrapper(
        f,
        encoding='ascii',
        errors='backslashreplace').read()

zip_file = 'enron1.zip'
data = []
with zipfile.ZipFile(zip_file, 'r') as zf:
    for file_info in zf.infolist():
        fname = file_info.filename
        if not fname.endswith('.txt'):
            continue
        if fname.endswith('Summary.txt'):
            continue
        content = read_file_from_zip(zf, fname)
        label = 'ham' if fname.startswith('enron1/ham/') else 'spam'
        data.append({'email_text': content, 'spam_or_ham': label})
                
emails_df = pd.DataFrame(data)
print(emails_df.sample(5))
db = sqlite3.connect('enron.sqlite')
emails_df.to_sql('enron',
                 db,
                 index=False,
                 if_exists='replace')
