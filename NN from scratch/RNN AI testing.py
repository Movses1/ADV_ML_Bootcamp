from text_processing.text_data_loading import load_data, text_to_embedding, embedding_to_text
import numpy as np
from model import Model
from layers import DenseLayer, Dropout, InpLayer, RNN, Conv2D
import pickle

train_data = load_data('text_processing/')
class_cnt = train_data[0].shape[1]
inp_letters = 10

Train = False
model = 0

if Train:
    model = Model([InpLayer(shape=np.array([inp_letters, class_cnt])),
                   # Conv2D(kernel_size=(3, 3), stride=(1, 1), filters=10, activation='relu', k_init='he_normal'),
                   RNN(neurons=100, activation='tanh', k_init='he_normal'),
                   DenseLayer(neurons=50, activation='relu', k_init='he_normal'),
                   RNN(neurons=100, activation='tanh', k_init='he_normal'),
                   DenseLayer(neurons=50, activation='relu', k_init='he_normal'),
                   DenseLayer(neurons=class_cnt, activation='softmax', k_init='glorot_uniform')],  # classification
                  loss='cce',
                  )
    model.fit_rnn(train_data, epochs=3000, batch_size=16, sequence_len=100, lr=1e-3)
    with open('model1.pkl', 'wb') as out_file:
        pickle.dump(model, out_file)
    print('saved')
else:
    with open(r"model1.pkl", "rb") as input_file:
        model = pickle.load(input_file)

inp = []
pred = 0
preds = np.zeros((inp_letters, class_cnt))


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
    preds[:-1] = preds[1:]
    preds[-1] = a
    return model.predict(np.array([preds]))


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