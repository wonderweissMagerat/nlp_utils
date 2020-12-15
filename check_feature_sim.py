import pickle
import json
import sys
import feature_similarity
import url_utils

def save_label_feature_info(path,label,feature,info):
    data = (label,feature,info)
    file=open(path,'wb')
    pickle.dump(data,file,0)


def get_bert_embedding(jd):
    url = 'http://bert-embed-4-doc.ha.svc.k8sc1.nb.com:8086'
    stitle = jd['stitle']
    seg_content = jd['seg_content']
    doc_id = 'test'
    info = {'content':{'stitle':stitle,'seg_content':seg_content,'doc_id':doc_id}}
    emb = json.loads(url_utils.post_url(url,info))['embedding']
    return emb


def load_getsim(path, key,query,top=10,sparse = False):
    (label,feature,info) = pickle.load(open(path,'rb'))
    for i in range(len(label)):
        if info[i][key]==query:
            print(info[i][key])
            
            if sparse:
                q_feature,q_label,q_info = feature.getcol(i),label[i],info[i]
            else:
                q_feature,q_label,q_info = feature[i],label[i],info[i]
    sort_dict = {}
    for i in range(len(label)):
        if sparse:
            cur_feature = feature.getcol(i)
            score = feature_similarity.cosine_distance_sparse(q_feature,cur_feature)
        else:
            cur_feature = feature[i]
            score = feature_similarity.cosine_distance(q_feature,cur_feature)
        if info[i][key]!= query:
            sort_dict[i] = score
    top_list = list(sorted(sort_dict.items(),key = lambda x:x[1],reverse = True))[:top]
    for item in top_list:
        index,score = item
        print(str(info[index][key])+'\t'+str(score)+'\t'+str(info[index]))

    
