import nltk
import pandas

# tokens = nltk.word_tokenize("KZ can you please shut the fuck up.")

# print("Parts of Speech: ", nltk.pos_tag(tokens))


filename = "./TA_restaurants_curated.csv"
data = pandas.read_csv(filename, sep=",", encoding="cp1252")


# print(len(data))
data = data.dropna(subset=["Reviews"])
# print(len(data))
# print(len(data))
data = data[data.Reviews != "[[], []]"]
# print(len(data))
# print(data["Reviews"].head(3)[0])

start, stop, step = 2,-2,1

data["Reviews"] = data["Reviews"].astype(str)
data["Reviews"] = data["Reviews"].str.slice(start, stop, step)
data["Reviews"] = data["Reviews"].str.split(", ")

def string_to_list(string_list):
	if len(string_list) > 2:
		#print(string_list)
		string_list[1] = string_list[1][:-1]
		string_list = string_list[:2]
		for i in range(len(string_list)):
			string_list[i] = string_list[i][1:-1]
		return string_list

	if len(string_list) > 1:
		print(string_list)

		string_list = string_list[:1]
		string_list[0] = string_list[0][:-1]
		for i in range(len(string_list)):
			string_list[i] = string_list[i][1:-1]
		# string_list[1] = string_list[1][1:] 
		print(string_list)

data["Reviews"] = data["Reviews"].apply(string_to_list)




# for i in range(len(data)):
# 	data["Reviews"][i] = data["Reviews"][i][1:-1]

# print(data["Reviews"].head(3)[0])








# File = open(fileName) #open file
# lines = File.read() #read all lines
# sentences = nltk.sent_tokenize(lines) #tokenize sentences
# nouns = [] #empty to array to hold all nouns

# for sentence in sentences:
#      for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
#          if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
#              nouns.append(word)