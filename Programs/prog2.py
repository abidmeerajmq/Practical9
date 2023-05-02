#!/usr/bin/env python
import sqlite3
import pandas
db = sqlite3.connect('enron.sqlite')
emails_df = pandas.read_sql('select * from enron', db)
print(emails_df.sample(5))
