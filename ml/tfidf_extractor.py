import sys
import numpy as np
from scipy.sparse import csr_matrix
import read_from_txt
import sparse_utils
import json

class TFIDF_Extractor:
    
    def __init__(self, idfpath='/mnt/nlp/big_sources/idf_unigram_top50w.dict',split = '\t'):
        self.idf = {}
        self.tfidf =[]
        self.vocab = {}
        data = read_from_txt.read_from_split(idfpath,split = split)
        for d in data:
            if d[0] not in self.idf:
                self.idf[d[0]] = float(d[1])
        self.vocab = dict(zip(list(self.idf.keys()),range(len(self.idf))))

    def fit(self, docs):
        #docs = [['a b c'],['d e f']]
        tfidf_bow = []
        for doc in docs:
            cur = {}
            for word in doc.split(' '):
                if word in self.idf:
                    cur[word] = cur.get(word,0)+self.idf[word]
            tfidf_bow.append(cur)
        self.tfidf = tfidf_bow
        return tfidf_bow

    def get_tfidf_bow(self):
        return self.tfidf

    def to_csr(self):
        return sparse_utils.dict_list_to_csr(self.tfidf, self.vocab)

    def to_array(self):
        return sparse_utils.dict_list_to_csr(self.tfidf, self.vocab).toarray()

if __name__ == '__main__':
    idf_path = '/mnt/nlp/big_sources/idf_unigram_top50w.dict'
    data = read_from_txt.read_from_split('./test_data')
    docs = []
    for doc in data:
        jd = json.loads(doc[1])
        docs.append(jd.get('seg_title',''))
    tfidf_tokenizer = TFIDF_Extractor(idf_path)
    tfidf_bow = tfidf_tokenizer.fit(docs)
    print(docs[1])
    print(tfidf_bow[1])
    tfidf_array = tfidf_tokenizer.to_array()
    print(sum(tfidf_array[1]))
    
