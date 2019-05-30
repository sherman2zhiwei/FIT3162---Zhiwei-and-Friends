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
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import xml.etree.ElementTree as et
from pathlib import Path
from time import time
from asp_agg_utils import _readXML, _add6PosFeautures, _flatten, _oneHotVectorize, _performance_measure, _clean_text

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub


# Keras
from keras import regularizers
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Convolution1D
from keras.layers import MaxPooling1D, Embedding, Dense, Dropout, GRU, Input, Lambda, concatenate
from keras.layers.core import Flatten, Activation
from keras.callbacks import EarlyStopping

# Gensim
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

"""
###################################################################
  __  __       _         _____                                     
 |  \/  |     (_)       |  __ \                                    
 | \  / | __ _ _ _ __   | |__) | __ ___   __ _ _ __ __ _ _ __ ___  
 | |\/| |/ _` | | '_ \  |  ___/ '__/ _ \ / _` | '__/ _` | '_ ` _ \ 
 | |  | | (_| | | | | | | |   | | | (_) | (_| | | | (_| | | | | | |
 |_|  |_|\__,_|_|_| |_| |_|   |_|  \___/ \__, |_|  \__,_|_| |_| |_|
                                          __/ |                    
                                         |___/                     
###################################################################
"""

if __name__ == '__main__':
    # This might be just for MacOS only
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    print(">>> Reading Data")
    data_parent_dir = "../Datasets/"
    filenames = ["Restaurants_Train.xml", "restaurants-trial.xml", "Restaurants_Test_Data_phaseB.xml"]
    datasets = list(map(lambda filename: _readXML(data_parent_dir + filename), filenames))
    print(">>> Dataset Structure (Rows, Columns)")
    print("Shape of train data:", datasets[0].shape)
    print("Shape of val data:", datasets[1].shape)
    print("Shape of test data:", datasets[2].shape)

    """
    #################################################################
      _____                                            _             
     |  __ \                                          (_)            
     | |__) | __ ___ _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
     |  ___/ '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
     | |   | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
     |_|   |_|  \___| .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
                    | |                                         __/ |
                    |_|                                        |___/ 
    #################################################################
    """

    NUM_WORDS = 100000 # Set maximum number of words to be embedded
    MAX_SENT_LEN = 65 # Set maximum length of a sentence

    train, val, test = datasets
    x_train = train["review"].apply(_clean_text)
    x_val = val["review"].apply(_clean_text)
    x_test = test["review"].apply(_clean_text)
    y_train = train["aspect"]
    y_val = val["aspect"]
    y_test = test["aspect"]
    unique_asp = pd.unique(_flatten(y_train))
    NUM_CLASSES = len(unique_asp)
    print("Type of aspects:", unique_asp)

    print(">>> Preparing Labels (One-Hot Encoding)")
    mlb = MultiLabelBinarizer(classes=[i for i in range(NUM_CLASSES)])
    le = LabelEncoder()
    le.fit(unique_asp)
    
    # Process labels into list of 1s and 0s
    y_train = _oneHotVectorize(y_train, mlb, le)
    y_val = _oneHotVectorize(y_val, mlb, le)
    y_test = _oneHotVectorize(y_test, mlb, le)
    print("Shape of y_train data:", y_train.shape)
    print("Shape of y_val data:", y_val.shape)
    print("Shape of y_test data:", y_test.shape)

    tokenizer = Tokenizer(num_words=NUM_WORDS, lower=False) # Define/Load Tokenize text function
    tokenizer.fit_on_texts(x_train) # Fit the function on the text
    word_index = tokenizer.word_index # Count number of unique tokens
    print(">>> Found %s unique tokens." % len(word_index))

    # Save tokenizer
    with open("tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Convert train, val and test to sequence
    sequences_train = tokenizer.texts_to_sequences(x_train)
    sequences_valid = tokenizer.texts_to_sequences(x_val)
    sequences_test = tokenizer.texts_to_sequences(x_test)

    # Limit size of train/val to 65 and pad the sequence
    x_train = pad_sequences(sequences_train, maxlen=MAX_SENT_LEN, padding="post")
    x_val = pad_sequences(sequences_valid, maxlen=MAX_SENT_LEN, padding="post")
    x_test = pad_sequences(sequences_test, maxlen=MAX_SENT_LEN, padding="post")

    print(">>> Before going through word embedding")
    print("Shape of x_train data:", x_train.shape)
    print("Shape of x_val data:", x_val.shape)
    print("Shape of x_test data:", x_test.shape)

    # Generate 6 POS features for each word in sentences from each dataset
    pos_train = _add6PosFeautures(train["review"])
    pos_val = _add6PosFeautures(val["review"])
    pos_test = _add6PosFeautures(test["review"])


    """
    ####################################################################################
     __          __           _   ______           _              _     _ _             
     \ \        / /          | | |  ____|         | |            | |   | (_)            
      \ \  /\  / /__  _ __ __| | | |__   _ __ ___ | |__   ___  __| | __| |_ _ __   __ _ 
       \ \/  \/ / _ \| '__/ _` | |  __| | '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` |
        \  /\  / (_) | | | (_| | | |____| | | | | | |_) |  __/ (_| | (_| | | | | | (_| |
         \/  \/ \___/|_|  \__,_| |______|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, |
                                                                                   __/ |
                                                                                  |___/ 
    ####################################################################################
    """

    emb_model_parent_dir = "../WordEmbedding/PretrainedEmbFiles/"
    emb_file = "GoogleNews-vectors-negative300.bin"
    # emb_file = "sentic2vec.csv" # Amazon
    # emb_file = "glove.6B.300d.txt"
    # emb_file = "crawl-300d-2M.vec"
    
    nn_model_parent_dir = "../SavedModels/"
    nn_model_file = nn_model_parent_dir + "googlenews_asp_agg_model.h5"
    isNNModelExist = Path(nn_model_file).is_file()
    print(">>> Check whether the network model exists")

    if isNNModelExist:
        print("Model exists...")
        model = load_model(nn_model_file)
        print("NN Model: Loaded!")
    else:
        print("Model not found...")
        print(">>> Loading pretrained embedding file")    
        
        # Google
        word_model = KeyedVectors.load_word2vec_format(emb_model_parent_dir + emb_file, binary=True)

        # Amazon
        # word_model = pd.read_csv(emb_model_parent_dir + emb_file, encoding = "ISO-8859-1", header=None)

        # Glove
        # word2vec_output_file = emb_model_parent_dir + "glove.6B.300d.word2vec"
        # if not Path(word2vec_output_file).is_file():
        #     glove2word2vec(emb_model_parent_dir + emb_file, word2vec_output_file)
        # word_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

        # Wiki Fasttext
        # word_model = KeyedVectors.load_word2vec_format(emb_model_parent_dir + emb_file, binary=False)

        print("Pretrained embedding model: Loaded!")

        EMBEDDING_DIM = 300
        VOCAB_SIZE = min(len(word_index)+1,(NUM_WORDS))
        embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

        for word, i in word_index.items():
            if i < VOCAB_SIZE:
                if word in word_model:
                    embedding_vector = word_model[word]
                    embedding_matrix[i] = embedding_vector
        del(word_model) # Delete word vectors
        

        """
        ####################################################
          ____        _ _     _   __  __           _      _ 
         |  _ \      (_) |   | | |  \/  |         | |    | |
         | |_) |_   _ _| | __| | | \  / | ___   __| | ___| |
         |  _ <| | | | | |/ _` | | |\/| |/ _ \ / _` |/ _ \ |
         | |_) | |_| | | | (_| | | |  | | (_) | (_| |  __/ |
         |____/ \__,_|_|_|\__,_| |_|  |_|\___/ \__,_|\___|_|
        ####################################################                                                                                                        
        """

        print(">>> Bulding Model")
        starttime = time() # start timer
        print("Start Time: ", starttime)

        NUM_FEATURE = 306
        FILTER_SIZE = [2, 3]
        NUM_FILTERS = [100, 50]
        POS_TAGS = 6

        pos = Input((MAX_SENT_LEN, POS_TAGS,))
        
        words = Input((MAX_SENT_LEN,))
        emb_layer = Embedding(VOCAB_SIZE,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        trainable=False,
                        input_length=MAX_SENT_LEN)(words)

        concat = concatenate([emb_layer, pos])
        conv1 = Convolution1D(NUM_FILTERS[0], FILTER_SIZE[0], border_mode="same", input_shape=(MAX_SENT_LEN, NUM_FEATURE))(concat)
        relu1 = Activation("relu")(conv1)
        pool1 = MaxPooling1D(pool_length=2)(relu1)
        conv2 = Convolution1D(NUM_FILTERS[1], FILTER_SIZE[1], border_mode="same")(pool1)
        relu2 = Activation("relu")(conv2)
        pool2 = MaxPooling1D(pool_length=2)(relu2)
        # gru = GRU(100,return_sequences=True)(pool2)
        # dropout = Dropout(0.25)(gru)
        # flat = Flatten()(dropout)
        flat = Flatten()(pool2)
        dense1 = Dense(512, input_shape=(MAX_SENT_LEN,))(flat)
        relu3 = Activation("relu")(dense1)
        # model.add(Dropout(0.5))
        out = Dense(NUM_CLASSES)(relu3)
        softmax = Activation("softmax")(out)
        model = Model(inputs=[words, pos], outputs=softmax)
        
        # Delete embedding matrix
        del(embedding_matrix) 

        # Generate model summary 
        model.summary()

        # Compile the model
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

        # Apply EarlyStopping
        callbacks = [EarlyStopping(monitor="val_loss", patience=2)]

        # Train the model
        history = model.fit([x_train, pos_train], y_train,
                            epochs=30,
                            batch_size=10,
                            callbacks=callbacks,
                            # validation_split=0.1
                            validation_data=([x_val, pos_val], y_val))
        print("Time used: ", time()-starttime)
        print(">>> Saving NN Model")
        model.save(nn_model_file)
        print("NN Model: Saved!")

    """
    #####################################################################################################
      __  __                                 _____           __                                          
     |  \/  |                               |  __ \         / _|                                         
     | \  / | ___  __ _ ___ _   _ _ __ ___  | |__) |__ _ __| |_ ___  _ __ _ __ ___   __ _ _ __   ___ ___ 
     | |\/| |/ _ \/ _` / __| | | | '__/ _ \ |  ___/ _ \ '__|  _/ _ \| '__| '_ ` _ \ / _` | '_ \ / __/ _ \
     | |  | |  __/ (_| \__ \ |_| | | |  __/ | |  |  __/ |  | || (_) | |  | | | | | | (_| | | | | (_|  __/
     |_|  |_|\___|\__,_|___/\__,_|_|  \___| |_|   \___|_|  |_| \___/|_|  |_| |_| |_|\__,_|_| |_|\___\___|
    #####################################################################################################                                                                                                     
    """

    print(">>> Performance Measurement")
    print("Train")
    _performance_measure(model, [x_train, pos_train], y_train)
    print("Validation")
    _performance_measure(model, [x_val, pos_val], y_val)
    print("Test")
    _performance_measure(model, [x_test, pos_test], y_test)