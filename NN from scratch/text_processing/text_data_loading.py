import numpy as np
import json

embeddings_dict = dict()


def _create_embeddings():
    f = open("text.txt", "r")
    st = set()

    while True:
        line = f.readline()
        if not line:
            break
        if line != '\n':
            for i in line[line.find(':') + 1:-1].lower():
                st.add(i)
    f.close()

    global embeddings_dict
    embeddings_dict = dict()
    for ind, i in enumerate(st):
        embeddings_dict[i] = ind

    out_file = open('embeddings.json', "w")
    json.dump(embeddings_dict, out_file)
    out_file.close()


def _load_embeddings():
    global embeddings_dict
    embeddings_dict = dict()
    f = open('embeddings.json', "r")
    embeddings_dict = json.load(f)
    f.close()


def load_data(new_embeddings=False):
    if new_embeddings:
        _create_embeddings()
    else:
        _load_embeddings()

    X = [[]]

    f = open("text.txt", "r")
    while True:
        line = f.readline()
        if not line:
            X[-1] = np.array(X[-1])
            break
        if line == '\n' and len(X[-1]) != 0:
            X[-1] = np.array(X[-1])
            X.append([])
        else:
            for i in line[line.find(':') + 2:-1].lower():
                X[-1].append(np.zeros(len(embeddings_dict) + 1))
                X[-1][-1][embeddings_dict[i]] = 1

            # adding the end of sentence embedding
            X[-1].append(np.zeros(len(embeddings_dict) + 1))
            X[-1][-1][-1] = 1
    f.close()

    return X


"""
data = load_data()
cnt = 0
for i in data:
    cnt += i.shape[0]
    print(i.shape)
print(cnt)
"""
