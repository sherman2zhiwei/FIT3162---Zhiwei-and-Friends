#!/bin/bash

echo "Downloading necessary libraries for Python ..."
pip install numpy
pip install sklearn
pip install tensorflow
pip install tensorflow_hub
pip install pickle-mixin
pip install pathlib
pip install nltk
pip install keras
pip install gensim

read -p "Do you want to download word embedding files? (y|n): " download_flag
declare -i flag=$(echo $download_flag | tr -s '[:upper:]' '[:lower:]')
if [ "$flag" = "y" ]
then
	echo "Downloading Word Embedding"
	echo "=== AmazonWE ==="
	wget "http://sentic.net/AmazonWE.zip"
	echo "=== GoogleNews ==="
	wget "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing"
	echo "=== Glove 6B ==="
	wget "http://nlp.stanford.edu/data/glove.6B.zip"
	echo "=== Glove 840B ==="
	wget "http://nlp.stanford.edu/data/glove.840B.300d.zip"
	echo "=== Fasttext Wikipedia ==="
	wget "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
fi

echo "Installing NPM related modules..."
npm install express
npm install socket.io
npm install http
npm install path
npm install python-shell
npm init
