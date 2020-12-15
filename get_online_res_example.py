import json
import sys
import fea_ext
import train_ml
import predict
import evaluate
import logging
import time


def exp_v1(conf):
    #setting
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'start')
    y,x,jds = fea_ext.v1(conf['train_path'])
    
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'data done')
    
    model = train_ml.fit_save(x,y,model_type = conf['model_type'],\
                         para = conf['para'],\
                        model_path = conf['model_path'])

    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'model done')


    for path in conf['dev_paths']:
        y,x,jds = fea_ext.v1(path)
        output_path = conf['res_path']+path.split('/')[-1]+'.res'
        py,py_pro,vocab = predict.predict_ml(conf['model_path'],x,y,\
                                             output_path = output_path,\
                                            jd = jds,\
                                             threshold=0.5,key=['stitle','url'])
        evaluate.get_prf(output_path,vocab)
    
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'end')

def exp_v2(conf):
    #setting
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'start')
    y,x,jds = fea_ext.v2(conf['train_path'])
    
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'data done')
    
    model = train_ml.fit_save(x,y,model_type = conf['model_type'],\
                         para = conf['para'],\
                        model_path = conf['model_path'])

    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'model done')


    for path in conf['dev_paths']:
        y,x,jds = fea_ext.v2(path)
        output_path = conf['res_path']+path.split('/')[-1]+'.res'
        py,py_pro,vocab = predict.predict_ml(conf['model_path'],x,y,\
                                             output_path = output_path,\
                                            jd = jds,\
                                             threshold=0.5,key=['stitle','url'])
        evaluate.get_prf(output_path,vocab)
    
    cur_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
    print(cur_time+' '+'end')


if __name__ == '__main__':
    #v2
    conf = {}
    conf['train_path'] = '../data/train/v1.train'
    conf['dev_paths'] = ['../data/train/v1.dev',\
     '../data/test/0711.test',\
     '../data/test/badcase.test',\
     '../data/test/quality_weekly.test',\
    '../data/test/topcheck.test']
    conf['model_path'] = '../model/v2.gbdt.model'
    conf['model_type'] = 'GBDT'
    conf['para'] = {'learning_rate':0.1,'n_estimators':100}
    conf['res_path'] = '../res/v2'
    
    exp_v2(conf)
    
    '''
    #v1
    conf = {}
    conf['train_path'] = '../data/train/v1.train'
    conf['dev_paths'] = ['../data/train/v1.dev',\
     '../data/test/0711.test',\
     '../data/test/badcase.test',\
     '../data/test/quality_weekly.test',\
    '../data/test/topcheck.test']
    conf['model_path'] = '../model/v1.gbdt.model'
    conf['model_type'] = 'GBDT'
    conf['para'] = {'learning_rate':0.1,'n_estimators':100}
    conf['res_path'] = '../res/'
    
    exp_v1(conf)
    '''
