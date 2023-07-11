import numpy as np
from NN_from_scratch.model import Model
from NN_from_scratch.layers import DenseLayer, InpLayer, MultiDense
from text_data_loading_orig import load_data

"""
    THIS MODULE IMPLEMENTS SKIP-GRAM ALGORITHM
    
    CLASS_CNT = INITIAL ONE HOT ENCODED SIZE
    OUT_CNT = NUMBER OF NEIGHBORING LETTERS (MUST BE EVEN or will be floored)
"""
train_data = load_data()
class_cnt = train_data[0].shape[-1]
out_cnt = 6

model = Model([InpLayer(shape=np.array([class_cnt])),
               #DenseLayer(neurons=70, activation='relu', k_init='he_normal'),
               DenseLayer(neurons=15, activation='linear', k_init='glorot_uniform'),
               #DenseLayer(neurons=100, activation='relu', k_init='he_normal'),
               MultiDense(out_cnt=out_cnt, neurons=class_cnt, activation='softmax', k_init='glorot_uniform')],
              loss='cce', optimizer='adagrad'
              )
combined_data = train_data[0]
for ind in range(1, len(train_data)):
    combined_data = np.append(combined_data, train_data[ind], axis=0)
    if len(combined_data)>1e5:
        break
combined_data = combined_data[:int(1e5)]

x = []
y = []
neighbor_cnt = out_cnt // 2
for i in range(neighbor_cnt, combined_data.shape[0] - neighbor_cnt):
    x.append(combined_data[i])
    y.append(np.append(combined_data[i - neighbor_cnt:i], combined_data[i + 1:i + neighbor_cnt + 1], axis=0))
x = np.array(x)
y = np.array(y)

print(x.shape, y.shape)

model.fit(x, y, epochs=25, batch_size=32, lr=1e-3)
