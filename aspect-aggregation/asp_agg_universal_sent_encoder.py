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
import numpy as np
import pandas as pd
import itertools
import pickle
import string
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import xml.etree.ElementTree as et
from pathlib import Path
from asp_agg_utils import _readXML, _flatten, _oneHotVectorize, _clean_text

# Tensorflow
import tensorflow as tf
import tensorflow_hub as hub

# Keras
from keras import backend as K
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
###############################################
  ______                _   _                 
 |  ____|              | | (_)                
 | |__ _   _ _ __   ___| |_ _  ___  _ __  ___ 
 |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
 | |  | |_| | | | | (__| |_| | (_) | | | \__ \
 |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/

###############################################
"""

def _performance_measure(model, data, label=None):
    """
    This function is similar to the performance measure in asp_agg_utils.py 
    but there is a difference about how the model can be loaded.
    :arg {model} - a trained Keras model
    :arg {data} - the input for model
    :arg {label} - the label for the input 
    :return - None
    """
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        model.load_weights("../SavedModels/use_asp_agg.h5")  
        preds = model.predict(data)

    processed_preds = []
    for i in range(len(preds)):
        pred = list(map(lambda val: 1 if val > 0.175 else 0, preds[i]))
        processed_preds.append(pred)
        
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

    train, val, test = datasets
    x_train = train["review"].apply(_clean_text).tolist()
    x_val = val["review"].apply(_clean_text).tolist()
    x_test = test["review"].apply(_clean_text).tolist()

    x_train = np.array(x_train, dtype=object)[:, np.newaxis]
    x_val = np.array(x_val, dtype=object)[:, np.newaxis]
    x_test = np.array(x_test, dtype=object)[:, np.newaxis]

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
    y_train = _oneHotVectorize(y_train, unique_asp, mlb, le)
    y_val = _oneHotVectorize(y_val, unique_asp, mlb, le)
    y_test = _oneHotVectorize(y_test, unique_asp, mlb, le)
    print("Shape of y_train data:", y_train.shape)
    print("Shape of y_val data:", y_val.shape)
    print("Shape of y_test data:", y_test.shape)

    print(">>> Bulding Model")

    use_model_parent_dir = "../SavedModels/"
    use_model_file = use_model_parent_dir + "use_asp_agg.h5"
    isUSEModelExist = Path(use_model_file).is_file()
    print(">>> Check whether the network model exists")

    if isUSEModelExist:
        print("Model exists...")
    else:
        print("Model not found...")
        print(">>> Getting universal sentence encoder")
        # Import the Universal Sentence Encoder's TF Hub module
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" # DAN
        # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #Transformer Model
        embed = hub.Module(module_url)
        EMBED_SIZE = 512

        # Prepare a function for keras lambda layer to map the input to universal sentence encoder
        def UniversalEmbedding(x):
            return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

        sents = Input((1,), dtype=tf.string)
        emb_layer = Lambda(UniversalEmbedding, output_shape=(EMBED_SIZE,))(sents)
        dense1 = Dense(256)(emb_layer)
        relu3 = Activation("relu")(dense1)
        out = Dense(NUM_CLASSES)(relu3)
        softmax = Activation("softmax")(out)
        model = Model(inputs=sents, outputs=softmax) 

        # Generate model summary 
        model.summary()

        # Compile the model
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])

        # Train the model
        with tf.Session() as session:
            K.set_session(session)
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            history = model.fit(x_train, y_train,
                                epochs=50,
                                batch_size=10,
                                # callbacks=callbacks,
                                validation_data=(x_val, y_val))

            use_model_file = "../SavedModels/use_asp_agg.h5"
            print(">>> Saving NN Model")
            model.save_weights(use_model_file)
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
    model = load_model(use_model_file)

    print(">>> Performance Measurement")
    print("Train")
    _performance_measure(model, x_train, y_train)
    print("Validation")
    _performance_measure(model, x_val, y_val)
    print("Test")
    _performance_measure(model, x_test, y_test)