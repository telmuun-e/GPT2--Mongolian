import re
import pickle

def clean(data):
    data = data.replace("\xa0", " ")
    data = re.sub(r'\n+', '\n', data)
    data = data.replace("\n", " ")
    data = re.sub(r'\t+', '\t', data)
    data = data.replace("\t", " ")
    data = re.sub(r" +", " ", data)
    data = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data)
    return data

def split(data):
    data = re.split("(\w+\.\s|\w+\?\s)", data)
    d = []
    for i in range(0, len(data)-1, 2):
        d.append("".join((data[i], data[i+1])))
    return d

def check_size(data):
    s = [i for i in data if 15 <= len(i) <= 700]
    return s

if __name__ == "__main__":

    with open(r"../data/raw_data/dnn_0.txt", "r") as r:
        data = r.read()
        r.close()

    data = clean(data)
    data = split(data)
    data = check_size(data)

    with open(r'../data/cleaned_data/clean_dnn', 'wb') as fp:
        pickle.dump(data, fp)