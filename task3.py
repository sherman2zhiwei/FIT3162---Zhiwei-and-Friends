import nltk
import pandas

tokens = nltk.word_tokenize("KZ can you please shut the fuck up.")

print("Parts of Speech: ", nltk.pos_tag(tokens))


filename = "./TA_restaurants_curated.csv"
data = pandas.read_csv(filename, sep=',')

print(data["Reviews"].head(3)[0])


# File = open(fileName) #open file
# lines = File.read() #read all lines
# sentences = nltk.sent_tokenize(lines) #tokenize sentences
# nouns = [] #empty to array to hold all nouns

# for sentence in sentences:
#      for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
#          if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
#              nouns.append(word)