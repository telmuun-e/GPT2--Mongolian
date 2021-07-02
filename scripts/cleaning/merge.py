import pickle

with open(r'../../data/cleaned_data/clean_wiki', 'rb') as fp:
    d_wiki = pickle.load(fp)
    
with open(r'../../data/cleaned_data/clean_ikon', 'rb') as fp:
    d_ikon = pickle.load(fp)
    
with open(r'../../data/cleaned_data/clean_dnn', 'rb') as fp:
    d_dnn = pickle.load(fp)

data = d_wiki + d_ikon + d_dnn
print(len(data))

with open(r'../../data/cleaned_data/clean_data', 'wb') as fp:
    pickle.dump(data, fp)

with open(r'../../data/cleaned_data/clean_data.txt', 'w') as fp:
    for i in data:
        fp.write(i + " ")