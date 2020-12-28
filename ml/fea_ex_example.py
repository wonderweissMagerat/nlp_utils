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
import online
from preprocess import Preprocessor_En
import rule
import load_vocab
from sklearn.feature_extraction.text import CountVectorizer

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
            cur_title = jd['stitle']
            if cur_title == None:
                continue
            cur_content = jd['seg_content']
            if cur_content == None:
                continue
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


def online_fea(path):
    jds = read_from_txt.read_from_json(path)
    stopwords = online.get_local_diction('../dict/stopword_en.txt')
    idf_dict = online.get_value_dict('../dict/idf.dict')
    embedding = online.get_embedding_dict('../dict/idf.embedding',300)
    cate_dict = online.get_local_diction('../dict/cate_dict')
    feature = []
    label = []
    data = []
    for jd in jds:
        try:
            cur_title = jd['stitle']
            cur_content = jd['seg_content']
            cur_url = jd['url']
            cur_cate = None
            cur_label = jd['label']
            if cur_label == 'Adult':
                cur_label = '1.0'
            else:
                cur_label = '0.0'
        except Exception as e:
            print(str(e)+'\t'+str(jd))
            continue
        cur_x = []
        cur_x.extend(online.get_embedding_feature(online.preprocess(cur_title,stopwords),embedding,300,idf_dict))
        cur_x.extend(online.get_embedding_feature(online.preprocess(cur_content,stopwords),embedding,300,idf_dict))
        cur_x.extend(online.get_embedding_feature(online.preprocess(cur_url,stopwords),embedding,300,idf_dict))
        cate_fea,_ = online.get_category_onehot(cur_cate,cate_dict)
        cur_x.extend(cate_fea)
        feature.append(cur_x)
        label.append(cur_label)
        data.append(jd)
    return label,feature,data

def rule_fea(path):
    jds = read_from_txt.read_from_json(path)
    title = []
    content = []
    feature = []
    label = []
    data = []
    
    for jd in jds:
        try:
            cur_title = jd['stitle']
            cur_content = jd['seg_content'].lower()
            cur_label = jd['label']
        except Exception as e:
            print(str(e)+'\t'+str(jd))
            continue
        title.append(cur_title)
        content.append(cur_content)
        label.append(cur_label)
        data.append(jd)

    return label,feature,data


def get_sentences(path):
    
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'start')
    jds = read_from_txt.read_from_json(path)
    title = []
    content = []
    feature = []
    label = []
    data = []
    sentence_splitter = Preprocessor_En()
    for jd in jds:
        try:
            cur_title = jd['stitle']
            cur_content = jd['seg_content']
            if cur_content == None or len(cur_content)==0:
                continue
            cur_label = jd['label']

        except Exception as e:
            print(str(e)+'\t'+str(jd))
            continue
        title.append(cur_title)
        content.append(sentence_splitter.sentence_split(cur_content))
        
        
        label.append(cur_label)
        data.append(jd)

    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'data loaded')
    

    return label,title,content,data

def get_danger_sent(label,title,content,data):
    content_label_sent = []
    all_sent = []
    all_label = []
    all_jd = []
    for index,cur_label in enumerate(label):
        if cur_label == 'Adult':
            all_label.append(cur_label)
            all_sent.append(title[index])
            all_jd.append(data[index])
            for sent in content[index]:
                sent_label,sent_note = rule.sent_rule(sent)
                sent_model_label,sent_model_note = rule.lr_onehot_sent(sent)
                if sent_label == 'Adult' or sent_model_label == 'Adult':
                    all_label.append('Adult')
                    all_sent.append(sent)
                    data[index]['note'] = data[index].get('note','')+sent_note
                    all_jd.append(data[index])
        else:
            all_label.append(cur_label)
            all_sent.append(title[index])
            all_jd.append(data[index])
            for sent in content[index]:
                all_label.append('Normal')
                all_sent.append(sent)
                data[index]['note'] = data[index].get('note','')
                all_jd.append(data[index])
    return all_label,all_sent,all_jd 


def one_hot_fea(input_path, token_path,is_train):
    jds = read_from_txt.read_from_json(input_path)
    content = []
    data = []
    label = []
    for jd in jds:
        try:
            cur_title = jd['stitle']
            if cur_title == None:
                continue
            cur_content = jd['seg_content']
            if cur_content==None:
                continue
            cur_label = jd['label']
        except:
            continue
        content.append(cur_title+' '+cur_content)
        label.append(cur_label)
        data.append(jd)
    vocab = load_vocab.load_path_index('../dict/rf_feaimp_top_32w.dict')
    if is_train:
        tokenizer = CountVectorizer(ngram_range=(1,3),lowercase=False,binary=True,min_df=3,vocabulary = vocab)
        tokenizer.fit(content)
        pickle.dump(tokenizer,open(token_path,'wb'),0)
    else:
        tokenizer = pickle.load(open(token_path,'rb'))
    feature = tokenizer.transform(content)
    return label,feature,data

if __name__ == '__main__':
    label,title,content,data = get_sentences('../data/train/v2.train')
    all_label,all_sent,all_jd = get_danger_sent(label,title,content,data)
    for index,label in enumerate(all_label):
        print(all_label[index]+'\t'+all_sent[index]+'\t'+all_jd[index].get('note','')+'\t'+all_jd[index]['url'])


    '''
    ems_path = '../dict/v3.v1.dev.embs'
    #label,all_ems,jds = v4('../data/train/v1.train',model_path='../dict/v1_4_train.tokenier.pcl',is_train=True)
    label,all_ems,jds = v3('../data/train/v1.dev')
    #print(label[0])
    #print(all_ems[0])
    check_feature_sim.save_label_feature_info(ems_path,label,all_ems,jds)
    #query = \
            #'https://www.bolde.com/porn-stars-filmed-erotic-scenes-boat-national-park/'
            #'https://www.inquisitr.com/6376995/stefflon-don-netted-string-swimsuit/'
    #check_feature_sim.load_getsim(ems_path,key='url',query=query,top=10,sparse=False)
    

    ems_path = '../dict/v3.quality_weekly.test.embs'
    label,all_ems,jds = v3('../data/test/quality_weekly.test')
    check_feature_sim.save_label_feature_info(ems_path,label,all_ems,jds)
    '''
