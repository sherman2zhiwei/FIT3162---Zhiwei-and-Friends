#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 02:23:50 2019

@author: Zhiwei and Friend(s)
"""

import flask
import numpy as np
import itertools
import tensorflow as tf
from model.data_utils import CoNLLDataset
from model.aspect_model import ASPECTModel
from model.config import Config

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def init():
    """
    This function is to load the trained Keras model.
    :return - None
    """
	# load the pre-trained tensorflow model
    global model,graph

    # create instance of config
    config = Config()

    # build model
    model = ASPECTModel(config)
    model.build()
    model.restore_session(config.dir_model)
    graph = tf.get_default_graph()

# Getting Parameters
def getParameters():
    """
    This function is to get the parameter(s) from request arguments (e.g. value of param in "http://www.example.com/?param=this is the input")
    :return - parameter(s)
    """
    parameters = []
    parameters.append(flask.request.args.get('sentence'))
    return parameters

# Cross origin support
def sendResponse(responseObj):
    """
    This function is to send the response packet to the local host server.
    :arg {responseObj} - unstructured response object
    :return - response packet
    """
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

def get_terms_from_IOBA(sent, tags):
    """
    This function will map the IOB aspect tag to the sentence given in order to extract the term(s).
    For example:
    sent = ["the", "food", "is", "nice"]
    tags = ["O", "B-A", "O", "O"]
    term extracted for this case is "food"
    :arg {sent} - list of words that represent the sentence given
    :arg {tag} - the IOB aspect tags for the sentence given
    :return - a customized string that concatenates all the terms extracted with the symbol of "|"
    """
    start = False
    terms = []
    for i, tag in enumerate(tags):
        if tag == "B-A":
            start = True
            terms.append([sent[i]])
        elif tag == "I-A" and start == True:
            terms[-1].append(sent[i])

        if start == True and tag == "O":
            start = False

    if len(terms) > 0:
        for i in range(len(terms)):
            terms[i] = " ".join(terms[i])

    return " | ".join(terms)

# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    """
    This function is the one to allow users to predict data through trained model. It will put it on hosted server's webpage
    :return - output in json
    """
    parameters = getParameters()
    input_sent = np.asarray(parameters).reshape(1)[0]

    input_sent = input_sent.strip().split(" ")

    # Prediction
    with graph.as_default():
    	prediction = model.predict(input_sent)

    # Map prediction tags to sentence in order to get the terms
    terms = get_terms_from_IOBA(input_sent, prediction)

    # When terms is empty string, it indicates there is no term found. 
    # Set it to None
    if terms == "":
    	terms = None
    	
    return sendResponse({"terms": terms})

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print("* Loading Tensorflow model and Flask starting server...please wait until server has fully started")
    init()
    app.run(threaded=True, port=5001)