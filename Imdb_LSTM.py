# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:51:36 2017

@author: leroy
"""

import urllib.request
import os
import tarfile

url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath = 'datas/aclImdb_v1.tar.gz'
if not os.path.isfile(filepath):
    results = urllib.request.urlretrieve(url, filepath)
    print("download:", results)
dataPath = 'datas/aclImdb'
if not os.path.isdir(dataPath):
    tfile = tarfile.open(filepath, 'r:gz')  
    result = tfile.extractall('datas/')
    
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import re 
def rm_tags(text):
    re_tag = re.compile(r"<[^>]+>")
    return re_tag.sub("", text)
import os
def read_files(filetype):
    path = "datas/aclImdb"
    file_list = [] 
    
    positive_path = path + "/" + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]
    
    negative_path = path + "/" + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]
    
    print("read", filetype, "files:", len(file_list))
    
    all_labels =([1]*12500 +[0]*12500)
    all_text = []
    for fi in file_list:
        with open(fi, encoding = "utf8") as file_input:
            all_text += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels, all_text

y_train, train_text = read_files("train")
y_test, test_text = read_files("test")

token = Tokenizer(num_words = 4000) 
token.fit_on_texts(train_text)

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

x_train = sequence.pad_sequences(x_train_seq, maxlen = 400)
x_test = sequence.pad_sequences(x_test_seq, maxlen = 400)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(Embedding(output_dim = 32, input_dim = 4000, input_length = 400))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(units = 512, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation = "sigmoid"))
print(model.summary())
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
train_history = model.fit(x_train, y_train, epochs = 10, batch_size = 200, validation_split = 0.2, verbose = 2)
scores = model.evaluate(x_test, y_test, verbose = 2)
scores[1]