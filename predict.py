import pickle
import json

def predict_ml(model_path,x,y,output_path,jd,threshold=0.5,key=[]):
    model = pickle.load(open(model_path,'rb'))
    class_ = model.classes_
    vocab = dict(zip(class_,range(len(class_))))
    predict_pro = model.predict_proba(x)
    output = open(output_path,'w')
    py = []
    py_pro = []
    for i in range(len(predict_pro)):
        for j in range(len(class_)):
            if predict_pro[i][j] >threshold:
                label = class_[j]
                proba = predict_pro[i][j]
                py.append(label)
                py_pro.append(proba)
                output.write('\t'.join([str(y[i]),str(label),str(proba)]))
                output.write('\t')
                output.write('\t'.join([jd[i][k] for k in key]))
                output.write('\n')
    return py,py_pro,vocab
    
