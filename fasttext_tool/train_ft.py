import fasttext
import read_from_txt
import sys
import json

def preprocess_context(label,content):
    context = content.split(' ')
    data_str = '__label__'+label+' '+' '.join(context)
    return data_str

def deal_train_data(input_path):
    data = read_from_txt.read_from_split(input_path)
    output = open(input_path+'.ft','w')
    for jd in data:
        try:
            cur_label = jd[0]
            cur_sent = jd[2]
        except:
            continue
        if len(cur_sent)>0:
            output.write(preprocess_context(cur_label,cur_sent)+'\n')


def train_ft_model(conf):
    para = conf['para']
    clf = fasttext.train_supervised(conf['train_path'],dim = para['dim'],\
        wordNgrams = para['wordNgrams'],minCount = para['minCount'])
    clf.save_model(conf['model_path'])
    return clf

def deal_test_data(input_path):
    data = read_from_txt.read_from_json(input_path)
    label = []
    title = []
    info = []
    for jd in data:
        try:
            cur_label = jd['label']
            cur_title = jd['stitle']
            #cur_content = jd['seg_content']
        except Exception as e:
            print(str(e))
            continue
        label.append(cur_label)
        title.append(cur_title)
        info.append(jd)
    return label,title,info

def ft_predict(model,content):
    res = model.predict([content],k=2)
    return res


if __name__ == '__main__':
    conf = {}
    train_path = '/home/services/zhaozhenyu/sensitive/v2/data/train/v1.train.check'
    conf['train_path'] = train_path+'.ft'
    conf['model_path'] = '/home/services/zhaozhenyu/sensitive/v2/model/ft_tst.ft.model'
    conf['dev_path'] = '/home/services/zhaozhenyu/sensitive/v2/data/train/v1.dev'
    conf['para'] = {'dim':100, 'wordNgrams':3,'minCount':5}
    deal_train_data(train_path)
    clf = train_ft_model(conf)
    label,title,data = deal_test_data(conf['dev_path'])
    #print(len(label))
    output = open(conf['dev_path']+'.ftres','w')
    for i,label in enumerate(label):
        res = ft_predict(clf,title[i])
        if res[0][0][0] == '__label__Adult':
            cur_label = 'Adult'
        else:
            cur_label = 'Normal'
        output.write(label+'\t'+str(res)+'\t'+title[i]+'\t'+data[i]['url']+'\n')
        
    
    
