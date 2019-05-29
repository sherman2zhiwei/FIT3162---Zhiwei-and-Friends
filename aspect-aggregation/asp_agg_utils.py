#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 02:23:50 2019

@author: Zhiwei and Friend(s)
"""

"""
###########################################
  _      _ _                    _          
 | |    (_| |                  (_)         
 | |     _| |__  _ __ __ _ _ __ _  ___ ___ 
 | |    | | '_ \| '__/ _` | '__| |/ _ / __|
 | |____| | |_) | | | (_| | |  | |  __\__ \
 |______|_|_.__/|_|  \__,_|_|  |_|\___|___/
                                           
###########################################
"""

import os
import re
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import xml.etree.ElementTree as et

# NLTK
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Keras
from keras.preprocessing.text import text_to_word_sequence

"""
###############################################
  ______                _   _                 
 |  ____|              | | (_)                
 | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
 |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
 | |  | |_| | | | | (__| |_| | (_) | | | \__ \
 |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/

###############################################
"""

def _readXML(filename):
    """
    This function is to read SemEval Dataset in XML format. Here, we only 7 columns, which are:
    ['review', 'term', 'termPolarity', 'startIndex', 'endIndex','aspect', 'aspectPolarity']
    :arg {filename} - the dataset file (e.g. "Restaurant_Train.xml")
    :return - pandas dataframe
    """
    table = []
    row = [np.NaN] * 7
    
    for event, node in et.iterparse(filename, events=("start", "end")):

        if node.tag == "text":
            row[0] = node.text
        elif node.tag == "aspectTerms" and event == "start":
            row[1] = []
            row[2] = []
            row[3] = []
            row[4] = []
        elif node.tag == "aspectTerm" and event == "start":
            row[1].append(node.attrib.get("term").replace("-", " ").replace("/", " "))
            row[2].append(node.attrib.get("polarity"))
            row[3].append(int(node.attrib.get("from")))
            row[4].append(int(node.attrib.get("to")))
        elif node.tag == "aspectCategories" and event == "start":
            row[5] = []
            row[6] = []
        elif node.tag == "aspectCategory" and event == "start":
            row[5].append(node.attrib.get("category"))
            row[6].append(node.attrib.get("polarity"))
        elif node.tag == "aspectCategories" and event == "end":
            table.append(row)
            row = [np.NaN] * 7

    dfcols = ['review', 'term', 'termPolarity', 'startIndex', 'endIndex','aspect', 'aspectPolarity']
    data = pd.DataFrame(table, columns=dfcols)
    data["review"] = data["review"].str.replace("-", " ") # remove hyphen
    data["review"] = data["review"].str.replace("/", " ") # remove slash
    return data
    
def _add6PosFeautures(sentences, max_sent_len = 65):
    """
    This function is specially made for add 6 POS tag features for the model we have trained.
    :arg {sentences} - list of sentences
    :arg {max_sent_len} - the maximum sentence length (by default it would be 65)
    :return - pos features for given list of sentences
    """
    le = LabelEncoder()
    pos_tags = ["CC","NN","JJ","VB","RB","IN"]
    le.fit(pos_tags)
    input_data = np.zeros((len(sentences), max_sent_len, len(pos_tags)))
    
    for i, sentence in enumerate(sentences):
        words = text_to_word_sequence(sentence)
        tags = nltk.pos_tag(words)
        sentence_len = len(tags)
        
        for j in range(max_sent_len):
            if j< sentence_len :
                curr_tag = tags[j][1][:2] # only see the first two letters
                if curr_tag in pos_tags:                    
                    index = (le.transform([curr_tag]))[0]
                    input_data[i][j][index] = 1

    return np.asarray(input_data)

# Flatten list of list
def _flatten(l):
    """
    This function will flatten a list of list to a list. (e.g. [[1],[2]] -> [1, 2])
    :arg {l} - a list of list
    :return - flattened list
    """
    return list(itertools.chain.from_iterable(l))

def _oneHotVectorize(df, asp_list, mlb, le):
    """
    This function acts as a vectorizer that turns a list of aspects into one-hot vector.
    However, it is modified to accommodate a multilabel pattern.
    :arg {df} - a dataframe (in this case, it would be our dataset)
    :arg {asp_list} - a unique aspect list (["service", "food", "price", "ambience", "anecdotes/miscellaneous"])
    :arg {mlb} - a multilabel binarizer (from module "sklearn")
    :arg {le} - a label encoder (from module "sklearn")
    :return - processed dataframe
    """
    df = df.apply(le.transform)
    df = mlb.fit_transform(df)
    return df

def _performance_measure(model, data, label=None):
    """
    This function is used to measure the model performance (Accuracy, Precision, Recall and F1) given the data and labels.
    Since the prediction made by the model is a list of list which consists of probabilities, a 0.175 threshold is set to extract labels.
    :arg {model} - a trained Keras model
    :arg {data} - the input for model
    :arg {label} - the label for the input 
    :return - None
    """
    preds = model.predict(data)

    processed_preds = []
    for i in range(len(preds)):
        pred = list(map(lambda val: 1 if val > 0.175 else 0, preds[i]))
        processed_preds.append(pred)
        
    # return the prediction if no label is provided.
    # as this would be in the case where users just want to see the output of model given their inputs 
    if label is None:
        return processed_preds

    test_label = processed_preds
    true_label = label

    total_pos = .0
    total_neg = .0
    tp = .0 # True Positive
    tn = .0 # True Negative
    for i in range(len(test_label)):
        for j in range(len(test_label[0])):
            if test_label[i][j] == 1:
                total_pos += 1
                if true_label[i][j] ==1:
                    tp +=1
            if test_label[i][j] == 0:
                total_neg += 1
                if true_label[i][j] ==0:
                    tn += 1

    fp = total_neg - tn # False Positive
    fn = total_pos - tp # False Negative
    precision = tp/(tp + fp)
    recall = tp/total_pos
    f1 = 2 * (precision * recall)/(precision + recall)
    acc = (tp + tn)/(total_pos + total_neg)

    print("Precision: " + str(round(precision, 4)))
    print("Recall: " + str(round(recall, 4))) 
    print("F1: " + str(round(f1, 4)))
    print("Accuracy: " + str(round(acc, 4)))

def _clean_text(text, stopwords=set(stopwords.words("english"))): #, lemmatizer=WordNetLemmatizer()):
    """
    This function is used for the preprocessing step, which will
    - convert text to lowercase
    - remove quotations surrounding the word (e.g. 'perks' -> perks)
    - handle some contraction of words (e.g. he's -> he is, can't -> cannot)
    - remove multiple consecutive spaces
    - remove the space that starts or ends in the sentence
    - remove stopwords
    (Note: the lemmatization has been done for this and we found that it did not provide a better result)
    :arg {text} - a string (sentence)
    :arg {stopwords} - a set of words 
    :return - preprocessed string
    """
    text = text.lower()
    text = re.sub(r"\'(\w*)\'", r"\1", text)
    text = re.sub(r"(he|she|it)\'s", r"\1 is", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    text = " ".join([w for w in word_tokenize(text) if not w in stopwords])
    # text = " ".join([_lemmatize(lemmatizer, w) for w in word_tokenize(text)])
    return text

# Scrap code for lemmatization
# def _get_wordnet_pos(nltk_pos_tag):

#     if nltk_pos_tag.startswith('V'):
#         return wordnet.VERB
#     elif nltk_pos_tag.startswith('N'):
#         return wordnet.NOUN
#     else:
#         return None

# def _lemmatize(lemmatizer, word):
#     nltk_pos_tag = nltk.pos_tag([word])[0][1]
#     wordnet_tag = _get_wordnet_pos(nltk_pos_tag)
#     if wordnet_tag is None:
#         return lemmatizer.lemmatize(word)
#     return lemmatizer.lemmatize(word, wordnet_tag)