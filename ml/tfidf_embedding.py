import sys
import json
import numpy as np

import embedding
from ml import tfidf_extractor
import read_from_txt



class TFIDFEmbeddingExtractor:
    def __init__(self, embedding_path='/mnt/nlp/big_sources/top50w.embedding', idf_path='/mnt/nlp/big_sources/idf_unigram_top50w.dict', default = None):
        self.embedding_class = embedding.Embedding(embedding_path)
        self.embedding = self.embedding_class.embedding
        self.tfidf_extractor = tfidf_extractor.TFIDF_Extractor(idf_path)

    def fit(self,docs, avg=True):
        #docs = [['a b c'],['d e f']]
        tfidf_bow = self.tfidf_extractor.fit(docs)
        tfidf_emb = []
        for doc in tfidf_bow:
            cur = np.zeros(self.embedding_class.dim)
            cur_sum = 0
            for key in doc:
                if key in self.embedding:
                    cur = cur+self.embedding[key]*doc[key]
                    cur_sum += doc[key]
            if avg and cur_sum > 0:
                cur = cur / cur_sum
            tfidf_emb.append(cur)

        self.tfidf_emb = np.array(tfidf_emb)
        return self.tfidf_emb


if __name__ == '__main__':
    data = read_from_txt.read_from_split('./test_data')
    docs = []
    for doc in data:
        jd = json.loads(doc[1])
        docs.append(jd.get('seg_title',''))
    tfidf_tokenizer = TFIDFEmbeddingExtractor()
    tfidf_bow = tfidf_tokenizer.fit(docs,avg=False)
    print(docs[1])
    print(tfidf_bow[1][:3])
    

