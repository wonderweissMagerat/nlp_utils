import sys
import json
import read_from_txt
import tfidf_embedding
import numpy as np
import parse_rule_file
import tfidf_extractor
import time
import check_feature_sim
import url_utils
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def v4(path,model_path,is_train):
    '''
    tfidf-ngram
    '''
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'start')
    jds = read_from_txt.read_from_json(path)
    title = []
    content = []
    feature = []
    label = []
    text = []
    title_rule = []
    content_rule = []
    data = []
    #parser = parse_rule_file.rule_parser('./rule_sys/vulgar/concept.txt','./rule_sys/vulgar/rule.txt')
    for jd in jds:
        try:
            cur_title = jd['stitle'].lower()
            cur_content = jd['seg_content'].lower()
            cur_label = jd['label']

        except Exception as e:
            print(str(e)+'\t'+str(jd))
            continue
        title.append(cur_title)
        content.append(cur_content)
        
        text.append((cur_title+' ')*3+cur_content)
        label.append(cur_label)
        data.append(jd)

    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'data loaded')
    if is_train:
        feature_tokenizer = TfidfVectorizer(ngram_range=(1,2),min_df=5)
    
        cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        print(cur_time+' '+'tfidf done')
        feature_tokenizer.fit(text)
        pickle.dump(feature_tokenizer,open(model_path,'wb'),0)
    else:
        feature_tokenizer = pickle.load(open(model_path,'rb'))
    all_ems = feature_tokenizer.transform(text)

    return label,all_ems,data




def get_bert_embedding(jd):
    url = 'http://bert-embed-4-doc.ha.svc.k8sc1.nb.com:8086'
    stitle = jd['stitle']
    seg_content = jd['seg_content']
    doc_id = 'test'
    info = {'content':{'stitle':stitle,'seg_content':seg_content,'doc_id':doc_id}}
    emb = json.loads(url_utils.post_url(url,info))['embedding']
    return emb





def v3(path):
    jds = read_from_txt.read_from_json(path)
    title = []
    content = []
    feature = []
    label = []
    data = []
    
    for jd in jds:
        try:
            cur_fea = get_bert_embedding(jd)
            cur_label = jd['label']
        except Exception as e:
            print(str(e)+'\t'+str(jd))
            continue
        feature.append(cur_fea)
        label.append(cur_label)
        data.append(jd)
    return label,feature,data




def v2(path):
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'start')
    jds = read_from_txt.read_from_json(path)
    title = []
    content = []
    feature = []
    label = []
    title_rule = []
    content_rule = []
    data = []
    parser = parse_rule_file.rule_parser('./rule_sys/vulgar/concept.txt','./rule_sys/vulgar/rule.txt')
    for jd in jds:
        try:
            cur_title = jd['stitle'].lower()
            cur_content = jd['seg_content'].lower()
            cur_label = jd['label']

            title_rule_res = parser.match(' '+cur_title+' ')
            cur_title_rule = ' '.join([k for k in title_rule_res if title_rule_res[k]])
            content_rule_res = parser.match(' '+cur_content+' ')
            cur_content_rule = ' '.join([k for k in content_rule_res if content_rule_res[k]])
        except Exception as e:
            print(str(e)+'\t'+str(jd))
            continue
        title.append(cur_title)
        content.append(cur_content)
        
        title_rule.append(cur_title_rule)
        content_rule.append(cur_content_rule)
        data.append(jd)
        label.append(cur_label)

    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'data loaded')

    feature_tokenizer = tfidf_embedding.TFIDFEmbeddingExtractor()
    title_ems = feature_tokenizer.fit(title)
    content_ems = feature_tokenizer.fit(content)
    all_ems = np.concatenate((title_ems,content_ems),axis=1)
    
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'tfidf_embedding done')

    rule_extractor = tfidf_extractor.TFIDF_Extractor(idfpath = '../dict/rule.dict',split = ' ')
    title_rule_fea = rule_extractor.fit(title_rule)
    title_rule_fea = rule_extractor.to_array()
    content_rule_fea = rule_extractor.fit(content_rule)
    content_rule_fea = rule_extractor.to_array()
    all_ems = np.concatenate((all_ems,title_rule_fea),axis=1)
    all_ems = np.concatenate((all_ems,content_rule_fea),axis=1)
    
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'rule end')

    return label,all_ems,data


def v1(path):
    jds = read_from_txt.read_from_json(path)
    title = []
    content = []
    feature = []
    label = []
    data = []
    
    for jd in jds:
        try:
            cur_title = jd['stitle'].lower()
            cur_content = jd['seg_content'].lower()
            cur_label = jd['label']
        except Exception as e:
            print(str(e)+'\t'+str(jd))
            continue
        title.append(cur_title)
        content.append(cur_content)
        label.append(cur_label)
        data.append(jd)
    feature_tokenizer = tfidf_embedding.TFIDFEmbeddingExtractor()
    title_ems = feature_tokenizer.fit(title)
    content_ems = feature_tokenizer.fit(content)
    all_ems = np.concatenate((title_ems,content_ems),axis=1)

    return label,all_ems,data


if __name__ == '__main__':
    
    ems_path = '../dict/v1_4.all.embs'
    #label,all_ems,jds = v4('../data/train/v1.train',model_path='../dict/v1_4_train.tokenier.pcl',is_train=True)
    #label,all_ems,jds = v2('../data/train/v1.train')
    #print(label[0])
    #print(all_ems[0])
    #check_feature_sim.save_label_feature_info(ems_path,label,all_ems,jds)
    query = \
            'https://www.bolde.com/porn-stars-filmed-erotic-scenes-boat-national-park/'
            #'https://www.inquisitr.com/6376995/stefflon-don-netted-string-swimsuit/'
    check_feature_sim.load_getsim(ems_path,key='url',query=query,top=10,sparse=True)

