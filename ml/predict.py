import pickle
import json

def predict_ml_binary(model_path,x,y,output_path,jd,threshold=('positive',0.5),key=[]):
    model = pickle.load(open(model_path,'rb'))
    class_ = model.classes_
    vocab = dict(zip(class_,range(len(class_))))
    predict_pro = model.predict_proba(x)
    output = open(output_path,'w')
    py = []
    py_pro = []
    for i in range(len(predict_pro)):
        if class_[0] == threshold[0]:
            if predict_pro[i][0]> threshold[1]:
                label = class_[0]
                proba = predict_pro[i][0]
            else:
                label = class_[1]
                proba = predict_pro[i][1]
        else:
            if predict_pro[i][1]> threshold[1]:
                label = class_[1]
                proba = predict_pro[i][1]
            else:
                label = class_[0]
                proba = predict_pro[i][0]
        py.append(label)
        py_pro.append(proba)
        output.write('\t'.join([str(y[i]),str(label),str(proba)]))
        output.write('\t')
        output.write('\t'.join([jd[i][k] for k in key]))
        output.write('\n')
    output.close()
    return py,py_pro,vocab
    
