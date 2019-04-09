import nltk
from preprocessing import *

def return_nouns(review):
    for word, pos in nltk.pos_tag(review):
        if pos in ['NN', "NNP"]:
            nouns.add(word)
    return nouns

if __name__ == "__main__":
    # data = readfile()
    # review_list = list(map(string_to_list, data))
    print(return_nouns("Just like home"))