from text_processing.text_data_loading import load_data, text_to_embedding, embedding_to_text
import numpy as np
from model import Model
from layers import DenseLayer, Dropout, InpLayer, RNN
import pickle

train_data = load_data('text_processing/')
class_cnt = train_data[0].shape[1]

Train = True
model = 0

if Train:
    model = Model([InpLayer(shape=np.array(class_cnt)),
                   RNN(neurons=100, activation='tanh', k_init='he_normal'),
                   DenseLayer(neurons=50, activation='relu', k_init='he_normal'),
                   RNN(neurons=100, activation='tanh', k_init='he_normal'),
                   DenseLayer(neurons=50, activation='relu', k_init='he_normal'),
                   DenseLayer(neurons=class_cnt, activation='softmax', k_init='glorot_uniform')],  # classification
                  loss='cce',
                  )
    model.fit_rnn(train_data, epochs=200, batch_size=16, sequence_len=100, lr=1e-3)
    with open('model1.pkl', 'wb') as out_file:
        pickle.dump(model, out_file)
else:
    with open(r"model1.pkl", "rb") as input_file:
        model = pickle.load(input_file)

while True:
    inp = input('Person: ')
    inp = text_to_embedding(inp.lower())
    # print(inp)
    sentence_end = np.zeros(inp.shape[1])
    sentence_end[-1] = 1
    inp = np.append(inp, [sentence_end], axis=0)

    print('AI: ', end='')
    pred = 0
    preds = []
    for i in inp:
        pred = model.predict(i)
    for i in range(100):
        pred = model.predict(pred)
        if np.argmax(pred[0]) == len(pred[0]) - 1:
            break
        preds.append(pred[0])

    preds = np.array(preds)
    if preds.size == 0:
        print('\n')
        continue

    preds = embedding_to_text(preds)
    for i in preds:
        print(i, end='')
    print('\n', preds.shape)
    print('\n')

