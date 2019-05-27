# Project Introduction
The focus of this project is on the aspect aggregation and aspect term extraction. It is mainly made for our FYP work.

***

# Libraries needed for FYP2
1. Aspect term extraction
  - os
  - numpy
  - sklearn
  - tensorflow
  - time
  - sys
  - logging

2. Aspect aggregation
  - os
  - numpy
  - sklearn
  - tensorflow
  - tensorflow_hub
  - itertools
  - pickle
  - string
  - pathlib
  - xml
  - nltk
  - keras
  - gensim
  - re

3. Word Embedding
  - AmazonWE (http://sentic.net/AmazonWE.zip)
  - GoogleNews (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
  - Glove (http://nlp.stanford.edu/data/glove.6B.zip, http://nlp.stanford.edu/data/glove.840B.300d.zip)
  - Fasttext Wiki (https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)

4. UI
  - json
  - lxml
  - npm (https://nodejs.org/en/download/)*
    - socket.io
    - fs
    - http
    - path
    - express
    - python-shell
(*: non-python related)

*** 

# Preliminary Check
Make sure your python is using the Anaconda path. You could check this by issue the command below to your terminal

`which python`

Example output:
![alt text](https://github.com/sherman2zhiwei/FIT3162---Zhiwei-and-Friends/images/check_python_path.png

UI preferred browser:
- Google Chrome

How to install npm on Windows:
- https://www.guru99.com/download-install-node-js.html

How to install npm on MacOS:
- https://blog.teamtreehouse.com/install-node-js-npm-mac

***

# Setup
Here, we wrote a bash script to facilitate you for the setup of all necessary libraries required. To run it, please issue this to your terminal:

`./setup_all.sh`

If you encounter any file permission problem from stopping you executing the bash script, please issue this to your terminal:

`chmod +x setup_all.sh`

***

# User Interface (UI)
The UI and its functionalities are created and connected using NodeJS framework and Python (Flask, Keras and Tensorflow). 
In order to let the users have a better experience on our project, we created 2 API links (one for Keras pre-trained model, another one for Tensorflow pre-trained model) hosted by 2 local servers to reduce the model loading time.

How to host 2 local servers for API links:
1. `python asp_agg_api.py` (in directory "FIT3162---Zhiwei-and-Friends")
2. `python asp_ext_api.py` (in directory "FIT3162---Zhiwei-and-Friends/aspect-extraction")

After the steps above are done, the following instruction is for you to start the main server which relies on the NodeJS framework. 
(The server-client network)

`node app.js` (in directory "FIT3162---Zhiwei-and-Friends")

### Note
Make sure port 5000, 5001 and 8080 are not in use. Otherwise, you might encounter issues of having busy ports. To kill those ports which are in use:

`kill $(lsof -t -i :YOUR_PORT_NUMBER)`
 
***
# Simple User Guide
Link: simple_user_guide.pdf

***

# Others
For more information about other issues you have faced, please contact us through e-mails:
- khoo0003@student.monash.edu (KZ)
- zwon0003@student.monash.edu (Zhiwei)
