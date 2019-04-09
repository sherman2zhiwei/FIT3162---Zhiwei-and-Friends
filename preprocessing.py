import nltk
import pandas

def readfile():
    filename = "./TA_restaurants_curated.csv"
    data = pandas.read_csv(filename, sep=",", encoding="cp1252")
    data = data.dropna(subset=["Reviews"])
    data = data[data.Reviews != "[[], []]"]

    start, stop, step = 2,-2,1

    data["Reviews"] = data["Reviews"].astype(str)
    data["Reviews"] = data["Reviews"].str.slice(start, stop, step)
    data["Reviews"] = data["Reviews"].str.split(", ")
    return data

def string_to_list(string_list):
    if len(string_list) > 2:
        string_list[1] = string_list[1][:-1]
        string_list = string_list[:2]
        for i in range(len(string_list)):
            string_list[i] = string_list[i][1:-1]
    else:
        string_list = string_list[:1]
        string_list[0] = string_list[0][:-1]
        for i in range(len(string_list)):
            string_list[i] = string_list[i][1:-1]
    return string_list
    
if __name__ == "__main__":
    data = readfile()
    review_list = list(map(string_to_list, data["Reviews"]))
    print(review_list[0])