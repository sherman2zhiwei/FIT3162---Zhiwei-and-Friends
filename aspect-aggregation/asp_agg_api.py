#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 02:23:50 2019

@author: Zhiwei and Friend(s)
"""

import flask
import numpy as np
import tensorflow as tf
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from asp_agg_utils import _add6PosFeautures, _performance_measure
from sklearn.preprocessing import LabelEncoder

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def init():
	"""
	This function is to load the trained Keras model.
	:return - None
	"""
	global model,graph
	# load the pre-trained Keras model
	model = load_model('../SavedModels/googlenews_asp_agg_model.h5')
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

# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
	"""
	This function is the one to allow users to predict data through trained model. It will put it on hosted server's webpage
	:return - output in json
	"""
	# Load the saved tokenizer
	with open('../tokenizer.pickle', 'rb') as handle:
	    tokenizer = pickle.load(handle)

	parameters = getParameters()
	input_sent = np.asarray(parameters).reshape(1)

	# Convert input to sequence
	MAX_SENT_LEN = 65
	# input_sent = "The food is cheap and the service is nice."
	data = tokenizer.texts_to_sequences(input_sent)
	seq_data = pad_sequences(data, maxlen=MAX_SENT_LEN, padding="post")

	# Generate POS features
	pos_data = _add6PosFeautures(input_sent)

	# Encode aspects as factors
	le = LabelEncoder()
	le.fit(["service", "food", "anecdotes/miscellaneous", "price", "ambience"])

	# Prediction
	x = [seq_data, pos_data]
	with graph.as_default():
		predicted_classes = _performance_measure(model, x)[0]

	prediction = []
	for i in range(len(predicted_classes)):
	    if predicted_classes[i] == 1:
	        prediction.append(le.inverse_transform([i])[0])

	prediction = " | ".join(prediction)
	return sendResponse({"aspects": prediction})

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print("* Loading Keras model and Flask starting server...please wait until server has fully started")
    init()
    app.run(threaded=True, port=5000)