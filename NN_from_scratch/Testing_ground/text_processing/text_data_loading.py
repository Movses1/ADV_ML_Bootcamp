import numpy as np
import json
import os

embeddings_dict = dict()


def transform_line(line):
    line = line.lower()
    line = line.replace('ã', 'a')
    line = line.replace('â', 'a')
    line = line.replace('ã', 'a')
    line = line.replace('€', 'e')
    line = line.replace('$', 's')
    line = line.replace('‚', ',')
    line = line.replace('“', '"')
    line = line.replace('”', '"')
    line = line.replace('’', "'")
    line = line.replace('‘', "'")
    line = line.replace('¦', '.')
    line = line.replace(':', '.')
    line = line.replace(';', '.')
    line = line.replace('&', 'and')
    for ind, i in enumerate(line):
        if i.isnumeric():
            line[ind] = ' '

    return ''.join([i if ord(i) < 128 else ' ' for i in line])


def _create_embeddings(path=''):
    f = open(path + 'text.txt', "r")
    st = set()

    while True:
        line = f.readline()
        if not line:
            break
        if line != '\n':
            for i in line[line.find(':') + 1:-1].lower():
                if not i.isnumeric():
                    st.add(i)
    f.close()

    global embeddings_dict, emb_ind
    embeddings_dict = dict()
    emb_ind = []

    weights1 = np.load('autoencoder_weights/weights1_2.0.npy')
    weights2 = np.load('autoencoder_weights/weights2_2.0.npy')
    bias1 = np.load('autoencoder_weights/biases1_2.0.npy')
    bias2 = np.load('autoencoder_weights/biases2_2.0.npy')

    for ind, i in enumerate(st):
        embeddings_dict[i] = ind
        emb_ind.append(i)
    emb_ind = np.array(emb_ind)
    embeddings_dict['\n'] = len(embeddings_dict)
    for k, v in embeddings_dict.items():
        v1 = np.zeros(len(embeddings_dict))
        v1[v] = 1
        v1 = v1 @ weights1 + bias1
        v1 = (v1 > 0)
        v1 = v1 @ weights2 + bias2
        embeddings_dict[k] = list(v1 * 1)

    print(embeddings_dict)
    out_file = open(path + 'embeddings1.json', "w")
    json.dump(embeddings_dict, out_file)
    out_file.close()
    print(embeddings_dict)


def _load_embeddings(path=''):
    global embeddings_dict
    embeddings_dict = dict()
    f = open(path + 'embeddings1.json', "r")
    embeddings_dict = json.load(f)
    f.close()


def text_to_embedding(text):
    x = []
    for i in text:
        if i.isnumeric():
            continue
        x.append(embeddings_dict[i])
    return np.array(x)


def embedding_to_text(emb):
    if emb.ndim == 1:
        emb = emb[np.newaxis, :]
    letters = []
    for e in emb:
        min_k = 0
        min_dist = np.inf
        for k, v in embeddings_dict.items():
            v = np.array(v)
            dist = np.linalg.norm(e - v)
            if dist < min_dist:
                min_dist = dist
                min_k = k
        letters.append(min_k)

    return letters


def load_data(path='', new_embeddings=False):
    if new_embeddings:
        _create_embeddings(path)
    else:
        _load_embeddings(path)

    X = [[]]

    cnt = 0
    lst_dir = 0
    if path == '':
        lst_dir = os.listdir()
    else:
        lst_dir = os.listdir(path)
    for i in lst_dir:
        """
        if cnt >= 100:
            break"""
        if i[-4:] != '.txt' or i == 'text.txt':
            continue
        f = open(path + i, "r", encoding='utf-8')
        while True:
            cnt += 1
            # print(cnt)
            line = f.readline()
            if not line:
                X[-1] = np.array(X[-1])
                X.append([])
                break
            if line == '\n' and len(X[-1]) != 0:
                X[-1] = np.array(X[-1])
                X.append([])
            else:
                sentence = text_to_embedding(line.lower())
                if sentence.shape[1] != 15:
                    print(sentence)
                for letter in sentence:
                    X[-1].append(letter)

        f.close()
    while len(X[-1]) == 0:
        X.pop()
    return X


"""data = load_data(new_embeddings=True)
cnt = 0
for i in data:
    cnt += i.shape[0]
    print(i.shape)
print(cnt)

print(embeddings_dict)
r = np.random.random((10, 35))
print(np.argmax(r, axis=1))
print(embedding_to_text(r))"""
