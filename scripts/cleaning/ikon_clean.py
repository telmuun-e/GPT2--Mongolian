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

    d = []

    for i in range(10):
        with open(r"../data/raw_data/ikon_{}.txt".format(i), "r") as r:
            data = r.read()
            r.close()

        print("ikon_{}.txt".format(i))

        data = clean(data)
        data = split(data)
        data = check_size(data)
        
        d.extend(data)
    
    with open(r'../data/cleaned_data/clean_ikon', 'wb') as fp:
        pickle.dump(d, fp)

