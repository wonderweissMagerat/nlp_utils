import sys
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer

import read_from_txt
import pickle

def lr_based_feature_select(y,x,vocab,output_path,model_type = 'LR'):
    if model_type == 'LR':
        model = linear_model.LogisticRegression()
        model.fit(x,y)
        importance = model.coef_[0]
    elif model_type == 'RF':
        model = RandomForestClassifier()
        model.fit(x,y)
        importance = model.feature_importances_
    else:
        raise SystemExit('current not support '+model_type)
    importance_map = {}
    for k in vocab:
        importance_map[k] = importance[vocab[k]]
    importance_map = sorted(importance_map.items(),key = lambda x:x[1],reverse = True)
    output = open(output_path,'w')
    for item in importance_map:
        output.write(str(item[0])+'\t'+str(item[1])+'\n')
    


    




def get_onehot_feature(content,tokenizer_path,ngram = (1,1),is_train = False):
    if is_train:
        tokenizer = CountVectorizer(ngram_range=ngram,lowercase=False,binary=True,min_df=3)
        tokenizer.fit(content)
        pickle.dump(tokenizer,open(tokenizer_path,'wb'),0)
    else:
        tokenizer = pickle.load(open(tokenizer_path,'rb'))
    feature = tokenizer.transform(content)
    vocab = tokenizer.vocabulary_
    return feature,vocab   

def get_context_from_json_file(path):
    jds = read_from_txt.read_from_json(path)
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
    return label,content,data


if __name__ == '__main__':
    label,content,data = get_context_from_json_file(sys.argv[1])
    feature,vocab = get_onehot_feature(content,sys.argv[2],ngram=(1,3),is_train=True)
    lr_based_feature_select(label,feature,vocab,sys.argv[3],model_type='RF')
