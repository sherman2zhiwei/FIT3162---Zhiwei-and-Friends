# Aspect extraction (Tensorflow)
## XML to IOB
Convert XML file to IOB file (Skip this as there are `.iob` files included in this repository)

```
python xmlToIOB.py
```

## Steps to build the model

### Word Embedding file required:
Step 1: Extract `glove.840B.300d.zip` to directory `aspect-extraction`

```
unzip glove.840B.300d.zip
```

Step 2: Build vocabularies from the data and extract trimmed glove vectors according to the configuration in `model/config.py`.

```
python build_data.py
```

Step 3: Train the model by issuing the command below

```
python train.py
```

3. To evaluate and interact with the model, do this
```
python evaluate.py
```

* Data iterators and utils are in `model/data_utils.py` and 

* Model with training/test procedures is in `model/aspect_model.py`

Once you have produced your data files, change the parameters in `config.py` like

```
# dataset
filename_train = "data/ABSA16_Restaurants_Train_SB1_v2_mod.iob"
filename_dev = "data/EN_REST_SB1_TEST_2016_mod.iob"
filename_test = "data/EN_REST_SB1_TEST_2016_mod.iob"
```

## Citation

Poria, S., Cambria, E. and Gelbukh, A., 2016. Aspect extraction for opinion mining with a deep convolutional neural network. Knowledge-Based Systems, 108, pp.42-49.

