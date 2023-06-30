import numpy as np
import tensorflow as tf

f = open("text.txt", "r")

speakers = set()
lens = np.zeros(100)
cnt = 0
line = f.readline()
while True:
    line = f.readline()
    if not line:
        break
    if line == '\n':
        continue

    spkr = line[:line.find(':')]
    speakers.add(spkr)
    lens[len(line) - len(spkr) - 1] += 1
    cnt += 1

print(cnt)
print(speakers)
print(lens)
