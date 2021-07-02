import re
import pickle

def clean(data):
    data = data.replace("\xa0", " ")
    data = re.sub(r'\n+', '\n', data)
    data = data.replace("\n", " ")
    data = re.sub(r" +", " ", data)
    data = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data)
    data = re.sub(r"↑", "", data)
    data = re.sub(r"\[\d+\]", "", data)
    return data

def split(data):
    data = re.split("(\w+\.\s)", data)
    d = []
    for i in range(0, len(data)-1, 2):
        d.append("".join((data[i], data[i+1])))
    q = []
    for i in d:
        q.extend(i.split("[засварлах | edit source] "))
    s = []
    for i in q:
        s.extend(i.split(", - "))
    return s

def check_size(data):
    s = [i for i in data if 15 <= len(i) <= 700]
    return s

if __name__ == "__main__":

    d = []

    for i in range(11):
        with open(r"../../data/raw_data/wiki_{}.txt".format(i), "r") as r:
            data = r.read()
            r.close()

        print("wiki_{}.txt".format(i))

        data = clean(data)
        data = split(data)
        data = check_size(data)
        
        d.extend(data)

    with open(r'../../data/cleaned_data/clean_wiki', 'wb') as fp:
        pickle.dump(d, fp)
