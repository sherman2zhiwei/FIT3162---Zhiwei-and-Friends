# Aspect Aggregation (Keras)

The model training files are:
- asp_agg_cnn_keras.py
- asp_agg_universal_sent_encoder.py
- basic_classifier.ipynb 

## To train these model, please issue this:
### Python File (.py)

```
python filename.py 
```

### Jupyter Notebook File (.ipynb)

Please refer to this link: https://jupyter.readthedocs.io/en/latest/install.html

### Note
If you wish to experience with different word embedding pre-trained models, uncomment the line below for that chosen model:

```python
  # emb_file = "GoogleNews-vectors-negative300.bin"
  # emb_file = "sentic2vec.csv" # Amazon
  # emb_file = "glove.6B.300d.txt"
  # emb_file = "crawl-300d-2M.vec"
```

Change the name of model if you want to skip the training process (the model name is in SavedModels):

```python
  nn_model_file = nn_model_parent_dir + "model_name.h5"
```

and also uncomment one of these according to the chosen model:

```python
  # Google
  # word_model = KeyedVectors.load_word2vec_format(emb_model_parent_dir + emb_file, binary=True)

  # Amazon
  # word_model = pd.read_csv(emb_model_parent_dir + emb_file, encoding = "ISO-8859-1", header=None)

  # Glove
  # word2vec_output_file = emb_model_parent_dir + "glove.6B.300d.word2vec"
  # if not Path(word2vec_output_file).is_file():
  #     glove2word2vec(emb_model_parent_dir + emb_file, word2vec_output_file)
  # word_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

  # Wiki Fasttext
  # word_model = KeyedVectors.load_word2vec_format(emb_model_parent_dir + emb_file, binary=False)
```
