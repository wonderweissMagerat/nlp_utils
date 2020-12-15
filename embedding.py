import sys
import json
import numpy as np
import time


class Embedding:
    def __init__(self, embedding_path='/mnt/nlp/big_sources/top50w.embedding',split = ' '):
        cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        print(cur_time+' '+'embedding load start')
        self.embedding = {}
        self.dim = None
        for lines in open(embedding_path):
            data = lines.strip().split(split)
            if len(data) == 2:
                self.dim == int(data[1])
                continue
            word = data[0]
            cur_embedding = np.array([float(i) for i in data[1:]])
            if self.dim == None:
                self.dim = len(cur_embedding)
            else:
                if self.dim != len(cur_embedding):
                    raise SystemExit('embedding dim not consist: before is '+str(self.dim)\
                        +' and current word('+word+') is'+str(len(cur_embedding)))
            if word not in self.embedding:
                self.embedding[word] = cur_embedding
        cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        print(cur_time+' '+'embedding load end')


if __name__ == '__main__':
    embedding = Embedding()
    print(embedding.embedding['love'])
