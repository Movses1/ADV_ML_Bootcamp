from text_processing.text_data_loading_orig import load_data, text_to_embedding, embedding_to_text
import numpy as np
from NN_from_scratch.model import Model
from NN_from_scratch.layers import DenseLayer, Dropout, InpLayer, RNN, Conv2D
import pickle

from text_processing.text_data_loading import _load_embeddings

"""
    THIS MODULE IMPLEMENTS RNN TRAINING

    CLASS_CNT = SIZE OF EACH LETTER EMBEDDING
    TRAIN = TRUE/FALSE (MODEL WILL BE SAVED OR LOADED)
    IN THE END YOU CAN SPEAK TO THE MODEL IN CONSOLE
"""
#_load_embeddings('text_processing/')
train_data = load_data('text_processing/')
class_cnt = train_data[0].shape[1]
inp_letters = 12
Train = True
model = 0

if Train:
    model = Model([InpLayer(shape=np.array([inp_letters, class_cnt, 1])),
                   #Conv2D(kernel_size=(1, 4), stride=(1, 2), filters=30, activation='relu', k_init='he_normal'),
                   RNN(neurons=(100, 50), activation='tanh', k_init='glorot_uniform'),
                   RNN(neurons=(100, 50), activation=('tanh', 'linear'), k_init='glorot_uniform'),
                   DenseLayer(neurons=100, activation='relu', k_init='he_normal'),
                   DenseLayer(neurons=100, activation='relu', k_init='he_normal'),
                   DenseLayer(neurons=class_cnt, activation='softmax', k_init='glorot_uniform')],  # classification
                  loss='cce', optimizer='adagrad',
                  )

    model.fit_rnn(train_data, epochs=200, batch_size=16, sequence_len=60, lr=1e-3)
    with open('model_orig.pkl', 'wb') as out_file:
        pickle.dump(model, out_file)
    print('saved')
else:
    with open(r"model_orig.pkl", "rb") as input_file:
        model = pickle.load(input_file)

inp = []
pred = 0
preds = np.zeros((1, inp_letters, class_cnt))


def human_input():
    global inp
    inp = input('Person: ')
    inp = text_to_embedding(inp.lower())
    sentence_end = np.zeros(inp.shape[1])
    sentence_end[-1] = 1
    inp = np.append(inp, [sentence_end], axis=0)


def make_preds(a):
    # a is the final datapoint
    global preds
    preds[0, :-1] = preds[0, 1:]
    preds[0, -1] = a
    return model.predict(preds)


while True:
    human_input()
    for i in inp:
        pred = make_preds(i)

    print('AI: ', end='')
    for i in range(100):
        print(embedding_to_text(pred[0]), end='')
        pred = make_preds(pred)

        # print(np.argmax(pred[0]), len(pred[0])-1)
        if np.argmax(pred[0]) == len(pred[0]) - 1:
            break

    print(' |\n')