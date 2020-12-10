import sys
import numpy as np

def run(path,label_vocab):
    t = 0 
    tp = 0 
    p = 0 
    cate_dict = label_vocab
    res = np.zeros((len(cate_dict),3))
    index_cate = zip(list(cate_dict.values()),list(cate_dict.keys()))
    for lines in open(path):
        data = lines.strip('\n').split('\t')
        for k in cate_dict:
            index = cate_dict[k]
            if k in data[0]:
                res[index][0]+=1
                t +=1 
                if k in data[1]:
                    res[index][1]+=1
                    tp +=1 
            if k in data[1]:
                res[index][2]+=1
                p +=1 
    output = open(path+'.prf','w')
    for cate in cate_dict:
        index = cate_dict[cate]
        cur  = res[index]
        r = cur[1] / float(cur[0]) if cur[0]!=0 else 0
        pre = cur[1] / float(cur[2]) if cur[2]!=0 else 0
        output.write(cate+'\t'+str(list(cur))+'\t'+str(r)+'\t'+str(pre)+'\n')
    r = tp / t if t!=0 else 0
    pre = tp / p if p!=0 else 0
    output.write('all\t'+str([t,tp,p])+'\t'+str(r)+'\t'+str(pre)+'\n')